"""Reactive NCCL baseline: no contact prediction, grants or drain barriers.

Rank zero alone decides wall-clock termination at sparse, deterministic
operation boundaries. Other ranks read the same decision before proceeding;
there is no READY/ACK/DONE handshake and no network collective for control.
Between checkpoints, training runs without shared control filesystem access.
"""
import json
import os
import time

from e3_schedule import ContactScheduler
from e3_startup import atomic_json, failure_status
from e3_broker import client_from_env


class ReactiveScheduler(ContactScheduler):
    def __init__(self, *args, comm_timeout=900, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode != 'off':
            raise ValueError('ReactiveScheduler requires mode=off')
        self.h = int(''.join(c for c in self.method if c.isdigit()) or '1')
        self.stop_interval = int(os.getenv('E3_REACTIVE_STOP_CHECK_OPS', '32'))
        self.timeout = max(float(comm_timeout)+60.,
                           float(os.getenv('E3_REACTIVE_STOP_TIMEOUT_S', '0')))
        self.poll = .02
        if self.stop_interval < 1:
            raise ValueError('E3_REACTIVE_STOP_CHECK_OPS must be positive')
        self.stop_dir = self.output / '.e3_control' / 'reactive_stop'
        self.in_batch = False
        self.next_step = 0 if self.h == 1 else self.h
        self.completed = 0
        self.broker=client_from_env()

    def _stop_check(self):
        path = self.stop_dir / f'checkpoint_{self.completed:09d}.json'
        if self.rank == 0:
            result = dict(run_id=self.run_id, operation=self.completed,
                          stop=time.time() >= self.end)
            if self.broker:self.broker.write('reactive_stop/'+path.name,result)
            else:atomic_json(path,result)
            return result['stop']
        deadline = time.monotonic()+self.timeout
        while True:
            if self.broker:result=self.broker.read('reactive_stop/'+path.name)
            else:
                try:
                    result = json.loads(path.read_text())
                except FileNotFoundError:
                    result = None
            if result is not None:
                if (result.get('run_id') != self.run_id
                        or result.get('operation') != self.completed):
                    raise RuntimeError('Stale reactive termination checkpoint')
                return bool(result['stop'])
            failed=(any(self.broker.read_many([f'failure/rank{i}' for i in range(4)]))
                    if self.broker else self.run_id and failure_status(self.output,self.run_id))
            if failed:
                raise RuntimeError('Reactive termination aborted: worker failure')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Reactive termination decision missing: {path}')
            time.sleep(self.poll)

    def gate(self, scope, step, bytes_, *, remaining_ops=1, tag='main'):
        if scope != 'collective' or step != self.next_step or remaining_ops < 1:
            raise RuntimeError('Unexpected reactive collective sequence')
        opid = f'{self.rank}-{tag}-{step}'
        ready = time.time()
        # Sparse STOP-only control, independent of the contact schedule.
        # The first checkpoint is at operation zero; all ranks subsequently
        # check the identical operation indices, never their own wall clocks.
        checked = self.completed % self.stop_interval == 0
        stop = self._stop_check() if checked else False
        now = time.time()
        fields = dict(comm_id=opid,current_step=step,scope=scope,
                      logical_bytes=bytes_,current_H=self.h,selected_H=self.h,
                      comm_ready_unix=ready,note='gate_tag='+tag,
                      rendezvous_mode='reactive_stop_check' if checked else 'none',
                      rendezvous_wait_s=now-ready if checked else 0.,
                      defer_duration_s=0.)
        if stop:
            self.record(**fields,scheduler_action='unavailable',
                        decision_reason='wall_budget',
                        communication_status='not_started',status='wall_budget')
            return False, opid
        self.record(**fields,scheduler_action='start',
                    decision_reason='baseline_no_schedule',
                    comm_start_unix=now,communication_status='started')
        return True, opid

    def finish_operation(self, comm_id, step):
        # Called after CUDA and optimizer/outer-model update. Pure local state.
        if step != self.next_step:
            raise RuntimeError('Reactive collective completion out of order')
        self.next_step += self.h
        self.completed += 1

    def close(self):
        if self.broker:self.broker.close()
        super().close()
