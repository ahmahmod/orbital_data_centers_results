"""E2 per-worker telemetry: E1_b-compatible columns and independent validation.
No collectives/barriers are introduced by monitoring or validation.
"""
import csv
import math
import pathlib
import statistics
import subprocess
import threading
import time

STEP_FIELDS = [
 'unix_s','elapsed_s','rank','step','loss','local_samples',
 'experiment','host','local_rank','epoch','unix_time_s','lr','global_samples_approx',
 'batch_samples','step_s','batch_wait_s','h2d_s','forward_s','backward_with_comm_s',
 'optimizer_s','other_s','collectives','collective_bytes_logical','collective_sum_host_s',
 'collective_span_host_s','collective_max_host_s','collective_failures',
 'gpu_memory_allocated_mb','samples_per_s','step_wall_s'
]
VALIDATION_FIELDS = [
 'experiment','method','rank','epoch','step','unix_time_s','elapsed_s',
 'val_loss','val_accuracy','val_samples','val_s','local_samples','global_samples_approx'
]
RESOURCE_FIELDS = [
 'rank','local_rank','host','unix_time_s','elapsed_s','gpu_util_pct',
 'gpu_memory_util_pct','gpu_mem_used_mib','gpu_power_w','rx_bytes_node',
 'tx_bytes_node','nic','status'
]

def writer(path, fields):
    fp = pathlib.Path(path).open('w', newline='', buffering=1)
    out = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
    out.writeheader()
    return fp, out


def sync_cuda(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_local(model, loader, device, torch, amp=False):
    """Same held-out data on every rank; no communication or model averaging."""
    was_training = model.training
    model.eval()
    n = correct = 0
    total_loss = 0.0
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                    logits = model(images)
                total_loss += torch.nn.functional.cross_entropy(
                    logits, labels, reduction='sum').item()
                correct += (logits.argmax(1) == labels).sum().item()
                n += labels.numel()
        sync_cuda(torch)
        return correct/max(1,n), total_loss/max(1,n), n, time.perf_counter()-start
    finally:
        model.train(was_training)


class ResourceSampler:
    """Samples GPU and NIC counters on the worker node; daemon stops on completion."""
    def __init__(self, output, rank, iface, start_unix, interval=2.0):
        self.path = pathlib.Path(output)/f'resources_rank{rank}.csv'
        self.rank = rank
        self.iface = iface
        self.start = start_unix
        self.period = max(0.5, float(interval))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f'e2-resources-{rank}', daemon=True)
    def start_sampling(self):
        self.thread.start()
    def close(self):
        self.stop_event.set()
        if self.thread.is_alive(): self.thread.join(timeout=7)
    def _run(self):
        import socket
        host = socket.gethostname()
        with self.path.open('w', newline='', buffering=1) as fp:
            w = csv.DictWriter(fp, fieldnames=RESOURCE_FIELDS)
            w.writeheader()
            while not self.stop_event.is_set():
                util = memutil = memused = power = None
                errors = []
                try:
                    p = subprocess.run([
                        'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                        '--format=csv,noheader,nounits', '-i', '0'],
                        capture_output=True, text=True, timeout=4)
                    if p.returncode: raise RuntimeError(p.stderr.strip())
                    fields = p.stdout.strip().splitlines()[0].split(',')
                    util,memutil,memused,power = [float(x.strip()) for x in fields[:4]]
                except Exception as exc:
                    errors.append('gpu:'+str(exc)[:80])
                def counter(field):
                    try:
                        return int((pathlib.Path('/sys/class/net')/self.iface/'statistics'/field).read_text())
                    except Exception as exc:
                        errors.append(field+':'+str(exc)[:40]);return ''
                now = time.time()
                w.writerow(dict(rank=self.rank,local_rank=0,host=host,unix_time_s=now,
                    elapsed_s=now-self.start,gpu_util_pct=util,gpu_memory_util_pct=memutil,
                    gpu_mem_used_mib=memused,gpu_power_w=power,
                    rx_bytes_node=counter('rx_bytes'),tx_bytes_node=counter('tx_bytes'),
                    nic=self.iface,status=';'.join(errors) if errors else 'ok'))
                self.stop_event.wait(self.period)
