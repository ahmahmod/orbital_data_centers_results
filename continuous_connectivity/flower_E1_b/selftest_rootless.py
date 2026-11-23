#!/usr/bin/env python3
"""Self-contained rootless socket pipeline test (no GPU, no root, no network)."""
import csv, json, os, pathlib, socket, subprocess, sys, tempfile, time
HERE = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='e1a_rootless_selftest_') as temp:
    root = pathlib.Path(temp)
    short = socket.gethostname().split('.')[0]
    (root/'worker_map.json').write_text(json.dumps({'workers': [
        {'rank':0,'pid':991234,'node':socket.gethostname(),'ip':'10.77.1.1'},
        {'rank':1,'pid':991235,'node':'mock-peer','ip':'10.77.1.2'}]}))
    bin_dir = root/'bin'; bin_dir.mkdir()
    fake_ss = bin_dir/'ss'
    fake_ss.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
f=Path(os.environ['E1_TEST_COUNTER'])
n=int(f.read_text()) if f.exists() else 0
n+=1024
f.write_text(str(n))
print('0 0 10.77.1.1:33333 10.77.1.2:44444 users:(("python",pid=991234,fd=10))')
print('    cubic rtt:2.0/1.0 bytes_sent:'+str(n)+' bytes_acked:'+str(n)+' bytes_received:'+str(n//2))
''')
    fake_ss.chmod(0o755)
    env = os.environ.copy()
    env.update(PATH=str(bin_dir)+os.pathsep+env.get('PATH',''),NET_IFACE='lo',E1_TEST_COUNTER=str(root/'counter.txt'))
    stop = root/'monitor_stop_test'
    p = subprocess.Popen([sys.executable,str(HERE/'socket_counter_watch.py'),str(root),'0.1','7','--stop-file',str(stop)],env=env)
    try:
        ready = root/f'monitor_ready_{short}'
        for _ in range(100):
            if ready.is_file(): break
            if p.poll() is not None: raise RuntimeError(f'monitor exited early with {p.returncode}')
            time.sleep(.1)
        assert ready.is_file(), 'monitor did not create ready file'
        time.sleep(2.8)
    finally:
        stop.touch()
        p.wait(timeout=10)
    assert p.returncode == 0, f'monitor exit {p.returncode}'
    samples = list(csv.DictReader((root/f'tcp_socket_samples_{short}.csv').open()))
    assert len(samples) >= 2, f'expected >=2 samples, got {len(samples)}'
    subprocess.run([sys.executable,str(HERE/'analyze_socket_counters.py'),str(root)],check=True)
    subprocess.run([sys.executable,str(HERE/'socket_monitor_health.py'),str(root),'--expected-hosts','1'],check=True)
    pairs = list(csv.DictReader((root/'tcp_rank_pair_summary.csv').open()))
    assert any(r['src_rank']=='0' and r['dst_rank']=='1' and int(r['tcp_payload_observed_delta_bytes'])>0 for r in pairs),pairs
    print('PASS: rootless monitor, graceful stop, socket attribution, directional byte totals, monitor health')
