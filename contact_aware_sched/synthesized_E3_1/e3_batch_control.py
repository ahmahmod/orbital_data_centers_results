"""Amortized contact admission. K counts collectives, never local steps H.

All ranks execute the same granted count, without shared filesystem control
operations inside the grant. Time estimates are conservative, not guarantees.
A grant cannot be revoked unilaterally. Unexpected NCCL failures are fatal.
"""
import csv
from collections import deque
import math
import os
import time

from e3_control import ContactScheduler
from e3_schedule import ContactScheduler as BaseScheduler


class BatchScheduler(ContactScheduler):
    def __init__(self, *args, comm_timeout=900, **kwargs):
        kwargs.setdefault('min_transfer',.5)
        super().__init__(*args, **kwargs)
        self.drain_timeout=max(self.timeout,float(comm_timeout)+60.)
        if self.mode != 'aware':
            raise ValueError('BatchScheduler is AWARE-only; use ReactiveScheduler for OFF')
        self.max_ops = int(os.getenv('E3_GRANT_MAX_OPS', '32'))
        self.local_step_budget = float(os.getenv('E3_LOCAL_STEP_BUDGET_S', '.25'))
        self.observed_factor = float(os.getenv('E3_OBSERVED_FACTOR', '1.5'))
        self.min_transfer = float(os.getenv('E3_MIN_TRANSFER_S', str(self.min_transfer)))
        if (self.max_ops < 1 or not math.isfinite(self.local_step_budget)
                or self.local_step_budget <= 0 or not math.isfinite(self.observed_factor)
                or self.observed_factor < 1 or not math.isfinite(self.min_transfer)
                or self.min_transfer < 0):
            raise ValueError('invalid batch scheduler parameters')
        self.h = int(''.join(c for c in self.method if c.isdigit()) or '1')
        self.batch = None
        self.sample_count=int(os.getenv('E3_OBSERVED_WINDOW','32'))
        self.sample_age=float(os.getenv('E3_OBSERVED_MAX_AGE_S','120'))
        self.quantile=float(os.getenv('E3_OBSERVED_QUANTILE','.9'))
        if self.sample_count<1 or not math.isfinite(self.sample_age) or self.sample_age<=0 or not 0<self.quantile<=1:
            raise ValueError('invalid rolling estimator settings')
        self.samples=deque(maxlen=self.sample_count)
        self.required_edges=None
        self.topology='full_mesh'
        self.batch_file = (self.output / f'e3_batch_rank{self.rank}.csv').open('w', newline='', buffering=1)
        self.batch_writer = csv.DictWriter(self.batch_file, fieldnames=[
            'rank','method','mode','batch_id','first_step','last_step','K','H',
            'ready_unix','grant_unix','contact_end_unix','done_unix','all_done_unix',
            'grant_wait_s','contact_wait_s','control_wait_excluding_contact_s',
            'drain_wait_s','estimated_operation_s','observed_max_operation_s',
            'budget_overrun','status','rate_bps','recent_estimate_s','topology'])
        self.batch_writer.writeheader()

    @property
    def in_batch(self):
        return self.batch is not None

    def set_topology(self, topology):
        edges=topology['required_edges']
        if not edges or any(a==b or a not in range(4) or b not in range(4) for a,b in edges):
            raise ValueError('invalid NCCL topology edges')
        self.required_edges=[(self.sats[a],self.sats[b]) for a,b in edges]
        if any((b,a) not in self.required_edges for a,b in self.required_edges):
            raise ValueError('missing reverse TCP ACK path')
        reached={self.sats[0]}
        for _ in range(4):
            reached.update(b for a,b in self.required_edges if a in reached)
        if reached!=set(self.sats):raise ValueError('disconnected NCCL topology')
        self.topology=topology['effective']
        self.window_cache.clear()

    def required(self, scope, peer=None):
        if scope=='collective' and getattr(self,'required_edges',None) is not None:
            return self.required_edges
        return super().required(scope,peer)

    def _recent(self, rows, at, rate, delay, full):
        """Worst rank's rolling nearest-rank quantile, only comparable
        contact regimes; samples expire even while waiting for a grant."""
        per_rank=[]
        for row in rows:
            values=sorted(s['elapsed'] for s in row['samples']
                          if -1<=at-s['stamp']<self.sample_age
                          and .8<=s['rate']/rate<=1.25
                          and abs(s['delay']-delay)<=max(.002,.25*delay)
                          and s['full']==full)
            if values:
                per_rank.append(values[max(0,math.ceil(self.quantile*len(values))-1)])
        return max(per_rank,default=0.),len(per_rank)==4

    def _plan(self, now, bytes_, limit, rows):
        if limit<1 or now>=self.end:return None
        # Reconsider a contact when an old outlier expires, rather than
        # waiting an additional orbital window solely due to stale data.
        expiries=sorted({s['stamp']+self.sample_age for row in rows for s in row['samples']
                         if s['stamp']+self.sample_age>now})
        for window in self.windows(self.required('collective')):
            earliest=max(now,self.start+window[0]+self.open_guard)
            end=min(self.end,self.start+window[1])
            if earliest>=end:continue
            rate,delay=window[2],window[4]
            for start in [earliest]+[x for x in expiries if earliest<x<end]:
                observed,calibrated=self._recent(rows,start,rate,delay,True)
                first_observed,_=self._recent(rows,start,rate,delay,False)
                cap=min(self.max_ops,limit)
                if not calibrated:cap=min(cap,1 if self.h==1 else 2)
                transfer=self.estimate(bytes_,window,'collective')
                unit=max(transfer+self.h*self.local_step_budget,observed*self.observed_factor)
                first=unit if self.h==1 else max(transfer+self.local_step_budget,
                                                  first_observed*self.observed_factor)
                budget=end-start-self.margin-self.launch_guard
                if budget<first:continue
                k=min(cap,1+max(0,int((budget-first)/unit)))
                return dict(start=start,end=end,estimate=unit,first_estimate=first,
                            K=k,rate=rate,delay=delay,recent_estimate=observed)
        return None

    def _state(self, folder, state, **details):
        if self.rank!=0:return
        values=' '.join(f'{k}={v}' for k,v in details.items())
        print(f'[E3] {folder.name} {state} {values}',flush=True)

    def _coordinate_batch(self, folder, bytes_, limit):
        self._state(folder,'WAIT_READY')
        rows = self._all(folder, 'ready', time.monotonic() + self.timeout)
        if any(row['bytes'] != bytes_ or row['limit'] != limit for row in rows):
            raise RuntimeError('Inconsistent collective payload or remaining operation count')
        last_state=None
        last_report=0.
        contact_wait = 0.
        attempt = 0
        while True:
            self._check_failure()
            now = time.time()
            plan = self._plan(now, bytes_, limit, rows)
            if plan is None:
                self._state(folder,'STOP',reason='no_feasible_contact_or_wall_budget')
                return dict(action='stop', reason='no_feasible_contact_or_wall_budget',
                            contact_wait=contact_wait)
            if plan['start'] > now:
                if last_state!='WAIT_CONTACT' or now-last_report>=30:
                    self._state(folder,'WAIT_CONTACT',elapsed=round(now-self.start,2),
                                next_start=round(plan['start']-self.start,2),
                                wait_s=round(plan['start']-now,2),estimate=round(plan['estimate'],3))
                    last_state='WAIT_CONTACT';last_report=now
                before = time.monotonic()
                time.sleep(min(.1, plan['start'] - now))
                contact_wait += time.monotonic() - before
                continue
            if not self._replay_ready(now - self.start):
                if last_state!='WAIT_REPLAY' or now-last_report>=30:
                    self._state(folder,'WAIT_REPLAY',elapsed=round(now-self.start,2))
                    last_state='WAIT_REPLAY';last_report=now
                before = time.monotonic()
                time.sleep(self.poll)
                contact_wait += time.monotonic() - before
                continue
            self._state(folder,'PREPARE',K=plan['K'],estimate=round(plan['estimate'],3))
            self._write(folder / f'prepare_{attempt}.json', **plan)
            self._write(folder / f'ack_{attempt}_0.json', ack=True)
            self._all(folder, f'ack_{attempt}', time.monotonic() + self.timeout)
            now = time.time()
            fresh = self._plan(now, bytes_, limit, rows)
            if fresh is None:
                return dict(action='stop', reason='no_feasible_contact_or_wall_budget',
                            contact_wait=contact_wait)
            if (fresh['start'] > now or fresh['end'] != plan['end']
                    or (self.mode == 'aware' and not self._replay_ready(now - self.start))):
                self._write(folder / f'retry_{attempt}.json', retry=True)
                attempt += 1
                continue
            # ACKs acknowledge readiness, not a fixed K. Only the final grant
            # authorizes operations; it may shrink if time was spent in ACKs.
            fresh['K'] = min(fresh['K'], plan['K'])
            self._state(folder,'GRANT',K=fresh['K'],estimate=round(fresh['estimate'],3),rate_mbps=round(fresh['rate']/1e6,1),contact_end=round(fresh['end']-self.start,2))
            return dict(action='grant', reason='predicted_window_feasible'
                        if self.mode == 'aware' else 'baseline_no_schedule',
                        contact_wait=contact_wait, **fresh)

    def _follow_batch(self, folder):
        attempt = 0
        deadline = time.monotonic() + max(0., self.end - time.time()) + self.timeout
        while True:
            self._check_failure()
            result = self._read(folder / 'decision.json')
            if result is not None:
                return result
            proposal = self._read(folder / f'prepare_{attempt}.json')
            if proposal is not None:
                self._write(folder / f'ack_{attempt}_{self.rank}.json', ack=True)
                result = self._wait(lambda: self._read(folder / 'decision.json')
                                    or self._read(folder / f'retry_{attempt}.json'),
                                    deadline, 'batch decision')
                if not result.get('retry'):
                    return result
                attempt += 1
            elif time.monotonic() >= deadline:
                raise TimeoutError(f'No batch decision: {folder}')
            else:
                time.sleep(self.poll)

    def gate(self, scope, step, bytes_, *, remaining_ops=1, tag='main'):
        if scope != 'collective':
            raise ValueError('BatchScheduler only supports collectives')
        opid = f'{self.rank}-{tag}-{step}'
        ready = time.time()
        wait = contact_wait = 0.
        if self.batch is None:
            folder = self.control / f'batch_{tag}_{step:09d}'
            self._write(folder / f'ready_{self.rank}.json', bytes=int(bytes_),
                        limit=int(remaining_ops), samples=list(self.samples))
            if self.rank == 0:
                result = self._coordinate_batch(folder, bytes_, int(remaining_ops))
                self._write(folder / 'decision.json', **result)
            else:
                result = self._follow_batch(folder)
            now = time.time()
            wait = now - ready
            contact_wait = min(wait, result['contact_wait'])
            if result['action'] == 'stop':
                self.record(comm_id=opid, current_step=step, scope=scope,
                            logical_bytes=bytes_, scheduler_action='unavailable',
                            communication_status='not_started', decision_reason=result['reason'],
                            comm_ready_unix=ready, defer_duration_s=contact_wait,
                            rendezvous_wait_s=max(0.,wait-contact_wait), note='gate_tag='+tag)
                self.batch_writer.writerow(dict(rank=self.rank,method=self.method,mode=self.mode,
                    batch_id=folder.name,first_step=step,K=0,H=self.h,ready_unix=ready,
                    grant_wait_s=wait,contact_wait_s=contact_wait,
                    control_wait_excluding_contact_s=max(0.,wait-contact_wait),status='stop'))
                return False, opid
            k = int(result['K'])
            if not 1 <= k <= remaining_ops:
                raise RuntimeError('Invalid granted count')
            self.batch = dict(folder=folder, first=step, next=step, bytes=bytes_,
                              tag=tag, K=k, remaining=k, result=result, ready=ready,
                              wait=wait, contact_wait=contact_wait, granted=now,
                              last_tick=time.monotonic(), peak=0.)
        b = self.batch
        if b['next'] != step or b['bytes'] != bytes_ or b['tag'] != tag or self.active:
            raise RuntimeError('Collective order mismatch inside batch')
        self.active[opid] = b['folder']
        fields = dict(comm_id=opid,current_step=step,scope=scope,logical_bytes=bytes_,
                      current_H=self.h,selected_H=self.h,comm_ready_unix=ready,
                      decision_reason=b['result']['reason'],note='gate_tag='+tag,
                      rendezvous_mode='batch_boundary' if step==b['first'] else 'batch_fast_path',
                      rendezvous_wait_s=max(0.,wait-contact_wait),defer_duration_s=contact_wait)
        if self.mode == 'aware':
            fields.update(expected_contact_start_unix=b['result']['start'],
                          expected_contact_end_unix=b['result']['end'],
                          predicted_contact_end_unix=b['result']['end'],
                          remaining_contact_s=b['result']['end']-time.time(),
                          estimated_sync_duration_s=b['result']['estimate'],
                          safety_margin_s=self.margin)
        if contact_wait > self.poll:
            self.record(**fields,scheduler_action='defer',communication_status='deferred')
        self.record(**fields,scheduler_action='start',comm_start_unix=time.time(),
                    communication_status='started')
        return True, opid

    def complete(self, comm_id, step, bytes_=0, **kwargs):
        """Local communication completion; no filesystem rendezvous here."""
        if comm_id not in self.active:
            raise RuntimeError('No active collective')
        # Skip the parent per-operation DONE barrier.
        BaseScheduler.complete(self, comm_id, step, bytes_, **kwargs)

    def finish_operation(self, comm_id, step):
        """After the optimizer/outer update and CUDA completion, drain at K."""
        b = self.batch
        if b is None or b['next'] != step or comm_id not in self.active:
            raise RuntimeError('Unexpected batch completion')
        self.active.pop(comm_id)
        tick = time.monotonic()
        elapsed = tick - b['last_tick']
        b['last_tick'] = tick
        # The first DiLoCo interval starts after local computation. It can
        # inform a floor but not calibrate full-round timing.
        self.samples.append(dict(stamp=time.time(),rate=b['result']['rate'],
                                 delay=b['result']['delay'],elapsed=elapsed,
                                 full=self.h==1 or step!=b['first']))
        b['peak'] = max(b['peak'], elapsed)
        b['remaining'] -= 1
        b['next'] += self.h
        if b['remaining']:
            return
        done = time.time()
        self._state(b['folder'],'DRAIN',K=b['K'])
        self._write(b['folder'] / f'done_{self.rank}.json', done=done)
        rows = self._all(b['folder'], 'done', time.monotonic() + self.drain_timeout)
        after = time.time()
        overrun = self.mode == 'aware' and max(row['done'] for row in rows) > b['result']['end']
        self.batch_writer.writerow(dict(rank=self.rank,method=self.method,mode=self.mode,
            batch_id=b['folder'].name,first_step=b['first'],last_step=step,K=b['K'],H=self.h,
            ready_unix=b['ready'],grant_unix=b['granted'],contact_end_unix=b['result']['end'],
            done_unix=done,all_done_unix=after,grant_wait_s=b['wait'],
            contact_wait_s=b['contact_wait'],control_wait_excluding_contact_s=max(0.,b['wait']-b['contact_wait']),
            drain_wait_s=after-done,estimated_operation_s=b['result']['estimate'],
            observed_max_operation_s=b['peak'],budget_overrun=int(overrun),status='drained',rate_bps=b['result']['rate'],
            recent_estimate_s=b['result']['recent_estimate'],topology=self.topology))
        self._state(b['folder'],'DONE',drain_s=round(after-done,3),overrun=int(overrun))
        self.record(comm_id=comm_id,current_step=step,scheduler_action='global_commit',
                    communication_status='committed',note='batch_drained')
        self.batch = None

    def close(self):
        self.batch_file.close()
        super().close()
