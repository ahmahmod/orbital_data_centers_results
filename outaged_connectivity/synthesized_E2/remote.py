"""Four-host Grid'5000 deploy-mode helpers. No oarsh: deployed nodes use ssh root."""
import json,os,pathlib,re,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parent

def run(argv, timeout=60, check=True):
    p=subprocess.run(argv, stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout)
    if check and p.returncode:raise RuntimeError(f'{shlex.join(map(str,argv))}: {p.returncode}: {p.stderr.strip()} {p.stdout.strip()}')
    return p.stdout.strip()

def hosts():
    nf=os.getenv('OAR_NODEFILE') or os.getenv('OAR_FILE_NODES')
    if not nf or not pathlib.Path(nf).is_file():raise RuntimeError('Run inside active OAR allocation')
    a=sorted(set(pathlib.Path(nf).read_text().split()))
    if len(a)!=4:raise RuntimeError(f'Expected exactly four physical hosts; got {a}')
    sites={site(h) for h in a}
    if len(sites)!=1:raise RuntimeError('kavlan-local requires one site')
    return a

def site(h):
    parts=h.split('.')
    if len(parts)<3:raise RuntimeError('Need site-qualified hostnames: '+h)
    return parts[1]

def profile(h):
    site_name=site(h)
    short=h.split('.')[0]
    found=[]
    for x in json.loads((ROOT/'sites.json').read_text()).get(site_name,[]):
        if short.startswith(x['cluster']+'-'):found.append(x)
    if len(found)!=1:raise RuntimeError(f'No safe NIC profile for {h}; add verified profile to sites.json')
    return found[0]

def data_nic(h):return profile(h)['iface']
def port_ref(h):return f"{h.split('.')[0]}-{profile(h)['eth']}.{site(h)}.grid5000.fr"

def ssh(h,command,timeout=60):
    if os.getenv('E2_REMOTE_MODE','deploy')=='deploy':
        return run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',f'root@{h}',command],timeout)
    return run(['oarsh',h,command],timeout)

def privileged(h,command,timeout=120):
    if os.getenv('E2_REMOTE_MODE','deploy')=='deploy':return ssh(h,command,timeout)
    wrapped=('if command -v sudo-g5k >/dev/null 2>&1; then sudo-g5k '+command+
             '; elif sudo -n true >/dev/null 2>&1; then sudo -n '+command+
             '; else echo "No sudo-g5k or sudo -n" >&2; exit 9; fi')
    return ssh(h,wrapped,timeout)

def remote_argv(h,command):
    if os.getenv('E2_REMOTE_MODE','deploy')=='deploy':return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',f'root@{h}',command]
    return ['oarsh',h,command]

def read_state():
    x=json.loads((ROOT/'kavlan_state.json').read_text())
    if x.get('oar_job_id')!=os.getenv('OAR_JOB_ID'):raise RuntimeError('KaVLAN state is for another reservation; run bootstrap.sh')
    if sorted(n['host'] for n in x['nodes'])!=hosts():raise RuntimeError('KaVLAN state hosts differ from OAR allocation')
    return x

def route_guard(host,iface,peer=None,src=None):
    q=shlex.quote
    ssh(host,f'ip link show dev {q(iface)} >/dev/null')
    rt=ssh(host,'ip -4 route show default')
    if 'dev '+iface in rt:raise RuntimeError(f'{host}: {iface} carries default management route')
    # Route to management frontend/SSH peer, not merely the default route.
    conn=ssh(host,'if [ -n "${SSH_CONNECTION:-}" ]; then ip -4 route get "${SSH_CONNECTION%% *}"; fi')
    if re.search(r'\bdev\s+'+re.escape(iface)+r'\b',conn):raise RuntimeError(f'{host}: experimental NIC used for SSH')
    if peer:
        rt=ssh(host,f'ip -4 route get {q(peer)}')
        if not re.search(r'\bdev\s+'+re.escape(iface)+r'\b',rt) or not re.search(r'\bsrc\s+'+re.escape(src)+r'\b',rt):
            raise RuntimeError(f'{host}->{peer} route bypasses experimental NIC: {rt}')
    return True
