#!/usr/bin/env python3
"""Rootless TCP_INFO sampling on each DDP host (no packet capture privileges).

Counters are cumulative per TCP connection; sampled observations can miss short-lived
connections. ss bytes_sent/received are TCP payload counters, NOT link-layer bytes.
"""
import argparse, csv, json, os, re, socket, subprocess, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('run_dir');p.add_argument('period',type=float);p.add_argument('duration',type=float)
p.add_argument('--stop-file',default='')
a=p.parse_args()
root=Path(a.run_dir);host=socket.gethostname().split('.')[0]
ready=root/f'monitor_ready_{host}'
stop=Path(a.stop_file) if a.stop_file else root/f'monitor_stop_{host}'
assert a.period >= .02 and a.duration > 0
fields=['unix_time_s','host','rank','pid','src_ip','src_port','dst_ip','dst_port','bytes_sent','bytes_received','bytes_acked','retrans','rtt_ms','socket_inode','sample_index']
meta_fields=['unix_time_s','host','sample_index','ss_duration_s','ss_returncode','connections_seen','owned_connections','worker_map_available','nic_rx_bytes','nic_tx_bytes','stderr']

# ss output has a connection heading followed by one or more TCP-info continuation lines.
# Parse IPv4 and optional [IPv6] syntax without relying on fixed ss columns.
def endpoint(value):
    m=re.match(r'^\[?((?:\d{1,3}\.){3}\d{1,3})\]?:(\d+)$',value)
    return (m.group(1),int(m.group(2))) if m else None

# ss output differs by iproute2 version and selected state filters. In some
# versions ESTAB is omitted and lines begin with "0 0 LOCAL PEER users:...";
# consequently fixed cols[3]/cols[4] silently discard every connection.
# Identify the local and remote TCP endpoints by syntax, not column number.
IPV4_ENDPOINT = re.compile(r'^(?:\[)?((?:\d{1,3}\.){3}\d{1,3})(?:\])?:(\d+)$')

def records(text):
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        cols = line.split()
        addresses = []
        for col in cols:
            m = IPV4_ENDPOINT.fullmatch(col)
            if m:
                addresses.append((m.group(1), int(m.group(2))))
        if len(addresses) >= 2:
            if current is not None:
                yield current
            current = {'src': addresses[0], 'dst': addresses[1],
                       'header': line.strip(), 'extra': ''}
        elif current is not None:
            # tcp_info may be on the continuation line, and owner metadata
            # may occur on either line depending on ss/iproute2 version.
            current['extra'] += ' ' + line.strip()
    if current is not None:
        yield current

def counter(text,name):
    # Avoid bytes_sent matching bytes_acked; numeric fields can be absent by kernel.
    m=re.search(r'(?<!\w)'+re.escape(name)+r':(\d+)(?!\w)',text)
    return int(m.group(1)) if m else ''

start=time.monotonic();index=0
with (root/f'tcp_socket_samples_{host}.csv').open('w',newline='',buffering=1) as sf, \
     (root/f'tcp_monitor_samples_{host}.csv').open('w',newline='',buffering=1) as mf:
    sw=csv.DictWriter(sf,fields);sw.writeheader()
    mw=csv.DictWriter(mf,meta_fields);mw.writeheader()
    ready.write_text(f'host={host} start_unix_s={time.time()}\n')
    while time.monotonic()-start<a.duration and not stop.exists():
        sample_start=time.monotonic();now=time.time();index+=1
        workers={};map_available=False
        try:
            wm=json.loads((root/'worker_map.json').read_text())['workers']
            workers={int(w['pid']):int(w['rank']) for w in wm if w['node'].split('.')[0]==host}
            map_available=True
        except (FileNotFoundError,ValueError,KeyError,TypeError):pass
        try:
            run=subprocess.run(['ss','-H','-t','-i','-e','-n','-p','state','established'],capture_output=True,text=True,timeout=8)
        except (OSError,subprocess.TimeoutExpired) as e:
            run=subprocess.CompletedProcess(args=['ss'],returncode=127,stdout='',stderr=str(e))
        n=0;owned=0
        if index == 1:
            # Diagnostic preview helps distinguish no sockets, unrecognized
            # format and absent process ownership without verbose logs.
            (root/f'ss_preview_{host}.txt').write_text(run.stdout[:4000] if run.stdout else '<no ss stdout>\n')
        if run.returncode==0:
            for r in records(run.stdout):
                n+=1
                pids={int(m) for m in re.findall(r'pid=(\d+)',r['header']+' '+r['extra'])}
                for pid in sorted(pids.intersection(workers)):
                    owned+=1
                    x=r['extra']
                    rtt=re.search(r'\brtt:([\d.]+)',x)
                    ino=re.search(r'\bino:(\d+)',r['header']+' '+r['extra'])
                    sw.writerow(dict(unix_time_s=round(now,6),host=host,rank=workers[pid],pid=pid,
                        src_ip=r['src'][0],src_port=r['src'][1],dst_ip=r['dst'][0],dst_port=r['dst'][1],
                        bytes_sent=counter(x,'bytes_sent'),bytes_received=counter(x,'bytes_received'),
                        bytes_acked=counter(x,'bytes_acked'),retrans=counter(x,'retrans'),
                        rtt_ms=float(rtt.group(1)) if rtt else '',socket_inode=ino.group(1) if ino else '',sample_index=index))
        def nic(direction):
            iface=os.environ.get('NET_IFACE','br0')
            try:return int(Path(f'/sys/class/net/{iface}/statistics/{direction}_bytes').read_text())
            except (OSError,ValueError):return ''
        mw.writerow(dict(unix_time_s=round(now,6),host=host,sample_index=index,
            ss_duration_s=round(time.monotonic()-sample_start,6),ss_returncode=run.returncode,
            connections_seen=n,owned_connections=owned,worker_map_available=int(map_available),
            nic_rx_bytes=nic('rx'),nic_tx_bytes=nic('tx'),stderr=run.stderr.strip()[:200]))
        sleep=max(0,a.period-(time.monotonic()-sample_start))
        if sleep:time.sleep(sleep)
(root/f'monitor_finished_{host}').write_text(f'host={host} end_unix_s={time.time()} samples={index}\n')
