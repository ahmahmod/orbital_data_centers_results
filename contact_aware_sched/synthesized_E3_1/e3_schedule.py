"""E3_1: contact-aware *communication timing*; H remains fixed (E3_2 later).

Reads the EXACT replay trace and the shared training start Unix timestamp.
Scheduling OFF is observational: returns immediately, logs the decision, and
never changes communication or optimization. No network calls in the planner.

Times use UTC epoch for cross-host joins and monotonic for duration measurement.
Not an end-to-end route oracle: collectives conservatively require all 12
directed worker pairs; gossip requires the outgoing edge; Flower needs
bidirectional server/client links for each of the four clients.
"""
import bisect,csv,json,math,os,pathlib,threading,time
from collections import defaultdict

FIELDS=['run_id','method','rank','comm_id','event_id','decision_unix','decision_elapsed_s',
 'current_step','sender_model_version','peer_rank','current_H','selected_H',
 'schedule_mode','scope','scheduler_action','decision_reason','status',
 'message_created_unix','comm_ready_unix','comm_start_unix','comm_complete_unix',
 'model_apply_unix','expected_contact_start_unix','expected_contact_end_unix',
 'next_contact_start_unix','next_contact_end_unix','predicted_contact_end_unix',
 'remaining_contact_s','estimated_sync_duration_s','safety_margin_s',
 'defer_duration_s','comm_blocked_s','logical_bytes','observed_tcp_bytes',
 'message_age_s','message_age_steps','receiver_step_at_apply',
 'outage_start_unix','outage_end_unix','worker_progress_during_outage',
 'recovery_time_s','rendezvous_wait_s','rendezvous_mode','communication_status','contact_trace_sha256','note']

class ContactScheduler:
    def __init__(self, trace, start_unix, duration, rank, method, output,
                 mode='off', margin=15., min_transfer=3., factor=2., run_id='',
                 rate_scale=1., clock_offset_s=0., rendezvous_lead_s=0.2):
        if mode not in ('off','aware'):raise ValueError('schedule mode off|aware')
        self.mode=mode;self.rank=int(rank);self.method=method
        self.start=float(start_unix);self.end=self.start+float(duration)
        self.margin=float(margin);self.min_transfer=float(min_transfer)
        self.factor=float(factor);self.rate_scale=float(rate_scale)
        self.offset=float(clock_offset_s);self.run_id=run_id
        self.rendezvous_lead_s=float(rendezvous_lead_s)
        if self.rendezvous_lead_s < 0:raise ValueError('rendezvous lead must be nonnegative')
        self.lock=threading.RLock();self.sequence=0
        self.output=pathlib.Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        import hashlib
        raw=pathlib.Path(trace).read_bytes()
        self.sha=hashlib.sha256(raw).hexdigest()
        rows=defaultdict(list)
        for r in csv.DictReader(raw.decode().splitlines()):
            key=(r['src'],r['dst'])
            rows[key].append((float(r['t_start_s']),float(r['t_end_s']),
                             int(r['available']),float(r['rate_bps'])*self.rate_scale,
                             float(r['prop_delay_ms'])/1000.))
        self.sats=[f'SAT-DEMO-{i+1:03d}' for i in range(4)]
        self.ranks={name:i for i,name in enumerate(self.sats)}
        self.edges={}
        self.window_cache={}
        for a in self.sats:
            for b in self.sats:
                if a==b:continue
                segments=sorted(rows.get((a,b),[]))
                if not segments or segments[0][0]>1e-6 or segments[-1][1]<float(duration)-1e-6:
                    raise ValueError(f'Incomplete trace for {a}->{b}')
                if any(abs(x[1]-y[0])>1e-5 for x,y in zip(segments,segments[1:])):
                    raise ValueError(f'Discontinuous trace for {a}->{b}')
                self.edges[a,b]=segments
        self.logfile=(self.output/f'e3_scheduler_decisions_rank{rank}.csv').open('w',newline='',buffering=1)
        self.writer=csv.DictWriter(self.logfile,fieldnames=FIELDS,extrasaction='ignore');self.writer.writeheader()
    def required(self,scope,peer=None):
        if scope=='peer':
            if peer is None:raise ValueError('peer required')
            return [(self.sats[self.rank],self.sats[int(peer)])]
        if scope=='flower':
            hub=self.sats[0]
            return [(x,hub) for x in self.sats if x!=hub]+[(hub,x) for x in self.sats if x!=hub]
        if scope=='collective':
            return list(self.edges)
        raise ValueError('invalid scope '+scope)
    def windows(self,required):
        cache_key=tuple(required)
        if cache_key in self.window_cache:return self.window_cache[cache_key]
        horizon=self.end-self.start
        bounds=sorted({0.0,horizon} | {
            t for edge in required for seg in self.edges[edge]
            for t in seg[:2] if 0.0 < t < horizon
        })
        result=[]
        for a,b in zip(bounds,bounds[1:]):
            if b<=a:continue
            middle=(a+b)/2
            matching=[]
            for edge in required:
                v=next((s for s in self.edges[edge] if s[0]<=middle<s[1]),None)
                if not v or v[2]!=1 or v[3]<=0:break
                matching.append(v)
            if len(matching)!=len(required):continue
            minimum_rate=min(s[3] for s in matching)
            maximum_delay=max(s[4] for s in matching)
            if result and abs(result[-1][1]-a)<1e-8 and result[-1][3]==minimum_rate:
                old=result[-1];result[-1]=(old[0],b,min(old[2],minimum_rate),minimum_rate,max(old[4],maximum_delay))
            else:result.append((a,b,minimum_rate,minimum_rate,maximum_delay))
        self.window_cache[cache_key]=result
        return result
    def estimate(self,bytes_,window,scope):
        rate=max(window[2],1.)
        parallel=3 if scope in ('collective','flower') else 1
        return max(self.min_transfer,self.factor*float(bytes_)*8*parallel/rate+2*window[4])
    def record(self,**fields):
        with self.lock:
            self.sequence+=1
            now=time.time()
            row=dict(run_id=self.run_id,method=self.method,rank=self.rank,
               event_id=self.sequence,decision_unix=now,decision_elapsed_s=now-self.start,
               schedule_mode=self.mode,contact_trace_sha256=self.sha)
            row.update(fields)
            self.writer.writerow(row)
            return self.sequence
    def rendezvous(self,scope,step,ready,tag='main'):
        """Coordination is through Grid'5000 shared filesystem, NOT impaired data links.
        All four ranks must agree on the SAME collective contact window.
        This overhead is intentionally measured and reported separately.
        """
        folder=self.output/'.e3_control'/f'{scope}_{tag}_{int(step):09d}'
        folder.mkdir(parents=True,exist_ok=True)
        my=folder/f'rank_{self.rank}'
        tmp=folder/f'.rank_{self.rank}_{os.getpid()}'
        tmp.write_text(str(ready))
        os.replace(tmp,my)
        timeout=float(os.environ.get('E3_RENDEZVOUS_TIMEOUT','180'))
        end=time.monotonic()+timeout
        from e3_startup import failure_status
        while not all((folder/f'rank_{i}').is_file() for i in range(4)):
            failed=failure_status(self.output,self.run_id) if self.run_id else {}
            if failed:
                raise RuntimeError(f'E3 rendezvous aborted: worker failure; '
                                   f'scope={scope}, step={step}, failures={failed}')
            if time.monotonic()>end:
                present=[i for i in range(4) if (folder/f'rank_{i}').is_file()]
                missing=sorted(set(range(4))-set(present))
                ages={}
                for i in present:
                    try:ages[i]=round(time.time()-float((folder/f'rank_{i}').read_text()),2)
                    except (OSError,ValueError):ages[i]='unreadable'
                raise TimeoutError(f'E3 shared-control rendezvous failed: '
                    f'{scope} step {step}; present={present}; missing={missing}; '
                    f'age_s={ages}; folder={folder}')
            time.sleep(.02)
        latest=max(float((folder/f'rank_{i}').read_text()) for i in range(4))
        return latest+float(os.environ.get('E3_RENDEZVOUS_LEAD_S',str(self.rendezvous_lead_s)))

    def gate(self,scope,step,bytes_,peer=None,version=None,comm_id=None,allow_until=None,created_unix=None,tag='main'):
        """Block only in aware mode. Return success. No network collectives used.
        Safe window requires ESTIMATED transfer + margin, not proof of completion.
        """
        now=time.time();ready=now
        common_ready=None
        if self.mode=='aware' and scope in ('collective','flower'):
            common_ready=self.rendezvous(scope,step,ready,tag=tag)
        after_rendezvous=time.time()
        if comm_id is None:comm_id=f'{self.rank}-{scope}-{step}-{int(now*1000000)}'
        keys=self.required(scope,peer);windows=self.windows(keys)
        current=((common_ready if common_ready is not None else now)-self.start)
        candidate=None
        for w in windows:
            t0,t1=w[:2]
            estimate=self.estimate(bytes_,w,scope)
            earliest=max(current,t0)
            if earliest+estimate+self.margin<=t1:
                candidate=(earliest,w,estimate)
                break
        kwargs=dict(comm_id=comm_id,current_step=step,sender_model_version=version,
                    rendezvous_wait_s=max(0,after_rendezvous-ready),
                    rendezvous_mode='shared_filesystem_control' if common_ready is not None else 'none',
                    peer_rank=peer if peer is not None else '',logical_bytes=bytes_,
                    scope=scope,comm_ready_unix=ready,message_created_unix=created_unix if created_unix is not None else ready,
                    estimated_sync_duration_s=candidate[2] if candidate else '',
                    safety_margin_s=self.margin,
                    remaining_contact_s=(candidate[1][1]-max(current,candidate[1][0])) if candidate else 0,
                    next_contact_start_unix=self.start+candidate[1][0] if candidate else '',
                    next_contact_end_unix=self.start+candidate[1][1] if candidate else '',
                    predicted_contact_end_unix=self.start+candidate[1][1] if candidate else '',
                    expected_contact_start_unix=self.start+candidate[1][0] if candidate else '',
                    expected_contact_end_unix=self.start+candidate[1][1] if candidate else '',
                    note='gate_tag='+str(tag),
                    current_H=int(''.join(c for c in self.method if c.isdigit()) or '1'),
                    selected_H=int(''.join(c for c in self.method if c.isdigit()) or '1'))
        if self.mode=='off':
            self.record(**kwargs,scheduler_action='start',decision_reason='baseline_no_schedule',
                        comm_start_unix=time.time(),communication_status='started',defer_duration_s=0)
            return True,comm_id
        if candidate is None:
            self.record(**kwargs,scheduler_action='unavailable',decision_reason='no_feasible_contact_before_end',
                        communication_status='not_started',status='no_feasible_contact')
            return False,comm_id
        earliest,w,estimate=candidate
        target=self.start+earliest
        if target>now:
            self.record(**kwargs,scheduler_action='defer',decision_reason='insufficient_contact_budget_or_outage',
                        communication_status='deferred',defer_duration_s=target-ready)
            while time.time()<target:
                if allow_until is not None and time.time()>=allow_until:return False,comm_id
                time.sleep(min(0.1,max(.001,target-time.time())))
        self.record(**kwargs,scheduler_action='start',decision_reason='predicted_window_feasible',
                    comm_start_unix=time.time(),defer_duration_s=max(0,time.time()-ready),
                    communication_status='started')
        return True,comm_id
    def complete(self,comm_id,step,bytes_=0,peer=None,version=None,status='completed',applied=False,note=''):
        self.record(comm_id=comm_id,current_step=step,peer_rank=peer if peer is not None else '',
                    sender_model_version=version,logical_bytes=bytes_,
                    comm_complete_unix=time.time() if status=='completed' else '',
                    model_apply_unix=time.time() if applied else '',
                    communication_status=status,scheduler_action='complete' if status=='completed' else 'failure',
                    note=note)
    def close(self):
        with self.lock:self.logfile.close()
