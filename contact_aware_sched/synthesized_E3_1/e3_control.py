"""Run-scoped, out-of-band collective admission for four synchronous workers.

Only rank zero selects a window. READY -> PREPARE -> ACK -> GRANT -> DONE.
No rank may enter NCCL on PREPARE, or independently cancel a GRANT.
A failed participant is fatal; the runner stops peers rather than reusing NCCL.
"""
import csv
import json
import os
import time
from pathlib import Path

from e3_startup import atomic_json, failure_status
from e3_broker import client_from_env
from e3_schedule import ContactScheduler as BaseScheduler


class ContactScheduler(BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control = self.output / '.e3_control' / 'nccl'
        self.timeout = float(os.getenv('E3_RENDEZVOUS_TIMEOUT', '180'))
        self.poll = float(os.getenv('E3_CONTROL_POLL_S', '.01'))
        self.open_guard = float(os.getenv('E3_CONTACT_OPEN_GUARD_S', '.25'))
        self.launch_guard = float(os.getenv('E3_LAUNCH_GUARD_S', '.25'))
        self.verify_replay = os.getenv('E3_VERIFY_REPLAY', '0') == '1'
        if min(self.timeout, self.poll) <= 0 or min(self.open_guard, self.launch_guard) < 0:
            raise ValueError('invalid E3 control timing')
        self.active = {}
        self.broker = client_from_env()

    def _write(self, path, **data):
        value = dict(run_id=self.run_id, **data)
        if self.broker:
            self.broker.write(path.relative_to(self.output/'.e3_control').as_posix(),value)
        else:
            atomic_json(path,value)

    def _read(self, path):
        if self.broker:
            data = self.broker.read(path.relative_to(self.output/'.e3_control').as_posix())
            if data is None:return None
        else:
            try:
                data = json.loads(path.read_text())
            except FileNotFoundError:
                return None
        if data.get('run_id') != self.run_id:
            raise RuntimeError(f'Stale control message: {path}')
        return data

    def _check_failure(self):
        if self.broker:
            failed = [x for x in self.broker.read_many([f'failure/rank{i}' for i in range(4)]) if x]
        else:
            failed = failure_status(self.output, self.run_id) if self.run_id else {}
        if failed:
            raise RuntimeError(f'Collective control aborted: {failed}')

    def _wait(self, fn, deadline, description):
        while True:
            self._check_failure()
            value = fn()
            if value is not None:
                return value
            if time.monotonic() >= deadline:
                raise TimeoutError(f'E3 control timeout: {description}')
            time.sleep(self.poll)

    def _all(self, folder, prefix, deadline):
        def read_all():
            paths = [folder / f'{prefix}_{rank}.json' for rank in range(4)]
            if self.broker:
                rows = self.broker.read_many([p.relative_to(self.output/'.e3_control').as_posix() for p in paths])
                if any(row is not None and row.get('run_id')!=self.run_id for row in rows):
                    raise RuntimeError('Stale control message')
            else:
                rows = [self._read(p) for p in paths]
            return rows if all(row is not None for row in rows) else None
        return self._wait(read_all, deadline, f'{folder.name}/{prefix}')

    def _window(self, earliest, bytes_):
        for window in self.windows(self.required('collective')):
            start = max(earliest, self.start + window[0] + self.open_guard)
            end = min(self.end, self.start + window[1])
            estimate = self.estimate(bytes_, window, 'collective')
            if start + self.launch_guard + estimate + self.margin <= end:
                return start, end, estimate
        return None

    def _replay_ready(self, elapsed):
        """Check flushed tc-agent records for the current trace segment per edge."""
        if not self.verify_replay:
            return True
        required=self.required('collective')
        if self.broker:
            keys=[f'replay/rank{self.ranks[src]}/{dst}' for src,dst in required]
            rows=self.broker.read_many(keys)
            for (src,dst),row in zip(required,rows):
                segment=next((s for s in self.edges[src,dst] if s[0]<=elapsed<s[1]),None)
                if (segment is None or segment[2]!=1 or not row
                        or row.get('run_id')!=self.run_id or row.get('available')!=1
                        or abs(row['t_start_s']-segment[0])>=1e-5):
                    return False
            return True
        for rank, src in enumerate(self.sats):
            destinations=[dst for a,dst in required if a==src]
            if not destinations:continue
            try:
                with (self.output / f'replay_rank{rank}.csv').open() as fp:
                    rows = list(csv.DictReader(fp))
            except FileNotFoundError:
                return False
            for dst in destinations:
                segment = next((s for s in self.edges[src, dst]
                                if s[0] <= elapsed < s[1]), None)
                if segment is None or segment[2] != 1:
                    return False
                matching = [row for row in rows if row.get('src') == src
                            and row.get('dst') == dst and row.get('applied_unix_s')
                            and abs(float(row['t_start_s']) - segment[0]) < 1e-5
                            and int(row['available']) == 1]
                if not matching:
                    return False
        return True

    def _coordinate(self, folder, bytes_, ready):
        if any(row['bytes'] != bytes_ for row in ready):
            raise RuntimeError('Collective payload mismatch across ranks')
        attempt = 0
        while True:
            self._check_failure()
            now = time.time()
            if now >= self.end:
                return dict(action='stop', reason='wall_budget')
            if self.mode == 'off':
                # Reactive baseline: same rendezvous and ACK protocol, no contact filter.
                candidate = (now, self.end, 0.)
            else:
                candidate = self._window(now, bytes_)
                if candidate is None:
                    return dict(action='stop', reason='no_feasible_contact_before_deadline')
            start, end, estimate = candidate
            if start > now:
                time.sleep(min(.1, start - now))
                continue
            if self.mode == 'aware' and not self._replay_ready(now - self.start):
                time.sleep(self.poll)
                continue
            self._write(folder / f'prepare_{attempt}.json',
                        start=start, end=end, estimate=estimate)
            self._write(folder / f'ack_{attempt}_0.json', ack=True)
            self._all(folder, f'ack_{attempt}', time.monotonic() + self.timeout)
            now = time.time()
            if now >= self.end:
                return dict(action='stop', reason='wall_budget')
            if self.mode == 'aware':
                # Recheck at commit, not at the original READY timestamp.
                fresh = self._window(now, bytes_)
                if (fresh is None or fresh[0] > now + 1e-6
                        or fresh[1] != end
                        or not self._replay_ready(now - self.start)):
                    self._write(folder / f'retry_{attempt}.json', retry=True)
                    attempt += 1
                    continue
                estimate = fresh[2]
            return dict(action='grant', start=now, end=end, estimate=estimate,
                        attempt=attempt, reason='predicted_window_feasible'
                        if self.mode == 'aware' else 'baseline_no_schedule')

    def gate(self, scope, step, bytes_, **kwargs):
        if scope != 'collective':
            raise ValueError('NCCL scheduler supports collective scope only')
        tag = kwargs.get('tag', 'main')
        folder = self.control / f'{tag}_{int(step):09d}'
        ready_at = time.time()
        opid = f'{self.rank}-{tag}-{step}'
        self._write(folder / f'ready_{self.rank}.json', bytes=int(bytes_), ready=ready_at)
        if self.rank == 0:
            # Coordinator must also acknowledge its own proposal.
            # _coordinate uses a separate ACK wait that includes rank zero.
            ready = self._all(folder, 'ready', time.monotonic() + self.timeout)
            result = self._coordinate(folder, bytes_, ready)
            self._write(folder / 'decision.json', **result)
        else:
            attempt = 0
            deadline = time.monotonic() + max(0., self.end - time.time()) + self.timeout
            while True:
                self._check_failure()
                result = self._read(folder / 'decision.json')
                if result is not None:
                    break
                proposal = self._read(folder / f'prepare_{attempt}.json')
                if proposal is not None:
                    self._write(folder / f'ack_{attempt}_{self.rank}.json', ack=True)
                    def outcome():
                        return self._read(folder / 'decision.json') or self._read(folder / f'retry_{attempt}.json')
                    result = self._wait(outcome, deadline, f'{folder.name}/decision')
                    if not result.get('retry'):
                        break
                    attempt += 1
                elif time.monotonic() >= deadline:
                    raise TimeoutError(f'No E3 grant: {folder}')
                else:
                    time.sleep(self.poll)
        now = time.time()
        fields = dict(comm_id=opid, current_step=step, scope=scope,
                      logical_bytes=bytes_, comm_ready_unix=ready_at,
                      rendezvous_mode='shared_filesystem_grant',
                      rendezvous_wait_s=now-ready_at, defer_duration_s=now-ready_at,
                      decision_reason=result['reason'], note='gate_tag='+tag)
        if result['action'] == 'stop':
            self.record(**fields, scheduler_action='unavailable',
                        communication_status='not_started', status='no_feasible_contact')
            return False, opid
        self.active[opid] = folder
        if self.mode == 'aware':
            fields.update(expected_contact_start_unix=result['start'],
                          expected_contact_end_unix=result['end'],
                          predicted_contact_end_unix=result['end'],
                          remaining_contact_s=result['end']-now,
                          estimated_sync_duration_s=result['estimate'],
                          safety_margin_s=self.margin)
        if now-ready_at > self.poll*3 and self.mode == 'aware':
            self.record(**fields, scheduler_action='defer', communication_status='deferred')
        self.record(**fields, scheduler_action='start', comm_start_unix=now,
                    communication_status='started')
        return True, opid

    def complete(self, comm_id, step, bytes_=0, **kwargs):
        # Caller must synchronize CUDA BEFORE reporting DONE. Record actual local
        # completion before waiting for peers so NCCL and control overhead differ.
        folder = self.active.pop(comm_id)
        super().complete(comm_id, step, bytes_, **kwargs)
        self._write(folder / f'done_{self.rank}.json', done=time.time())
        self._all(folder, 'done', time.monotonic() + self.timeout)
        self.record(comm_id=comm_id, current_step=step,
                    scheduler_action='global_commit', communication_status='committed')

    def close(self):
        if self.broker:self.broker.close()
        super().close()
