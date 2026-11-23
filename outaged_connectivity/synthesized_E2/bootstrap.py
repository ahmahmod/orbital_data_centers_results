#!/usr/bin/env python3
"""Deploy 4 workers and select isolated KaVLAN or safer shared-NIC selective tc.

Auto mode prefers KaVLAN when every node has a separate recognized KaVLAN port.
Otherwise it records management-NIC IPs for peer-IP selective shaping; DOES NOT
modify that NIC here. Never blindly assign management switch ports to a VLAN.
"""
import argparse,json,os,re,shlex,time
from remote import ROOT,hosts,run,ssh,privileged,profile,port_ref,route_guard
from network_select import choose_network,default_interface,interface_ip,validate_shared_routes


def kavlan_command(nodes,*args,timeout=90):
    from shutil import which
    job=os.environ.get('OAR_JOB_ID')
    if not job:raise RuntimeError('Missing OAR_JOB_ID')
    site=nodes[0].split('.')[1]
    command=['kavlan',*map(str,args),'-j',job]
    if which('kavlan'):
        return run(command,timeout=timeout)
    frontend=f'f{site}.{site}.grid5000.fr'
    return run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',frontend,shlex.join(command)],timeout=timeout)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--deploy',action='store_true')
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--environment',default=os.getenv('ENVIRONMENT','debian12-x64-nfs'))
    ap.add_argument('--network-mode',choices=['auto','kavlan','selective_tc'],default=os.getenv('NETWORK_MODE','auto'))
    ap.add_argument('--skip-gpu-check',action='store_true')
    a=ap.parse_args();nodes=hosts();print('Allocated hosts:',*nodes,sep='\n  ',flush=True)
    if a.deploy:
        os.environ['E2_REMOTE_MODE']='deploy'
        nf=os.environ.get('OAR_NODEFILE') or os.environ.get('OAR_FILE_NODES')
        print('Deploying',a.environment,flush=True)
        run(['kadeploy3','-e',a.environment,'-k','-f',nf],timeout=2400)
        (ROOT/'.deployment_job').write_text(str(os.getenv('OAR_JOB_ID','')))
    for h in nodes:
        print('Inspect',h,ssh(h,'hostname; ip -br link; ip -4 route show default'),flush=True)
        if not a.skip_gpu_check:
            try:print('GPU',h,ssh(h,'nvidia-smi --query-gpu=name --format=csv,noheader',timeout=25),flush=True)
            except RuntimeError as exc:print('WARNING: GPU runtime not ready:',h,exc,flush=True)
    def lookup(ref):
        return kavlan_command(nodes,'-g','-m',ref)
    selected,reports=choose_network(nodes,lookup)
    if a.network_mode=='kavlan' and selected!='kavlan':
        raise RuntimeError('Forced KaVLAN requires a safe separate, recognized data port on ALL four hosts. Details: '+json.dumps(reports))
    if a.network_mode=='selective_tc':selected='selective_tc'
    print('SELECTED NETWORK BACKEND:',selected,flush=True)
    for r in reports:print(json.dumps(r),flush=True)
    if not a.apply:
        print('READ ONLY. To write network state, run with --apply.')
        return
    mapped=[];vlan=None
    if selected=='kavlan':
        vlan=kavlan_command(nodes,'-V')
        if not re.fullmatch(r'[0-9]+',vlan) or int(vlan) not in (1,2,3):
            raise RuntimeError('Expected allocated KaVLAN-local ID 1, 2 or 3; got '+str(vlan))
        # Preflight of ALL ports was completed by choose_network before writing any membership.
        kavlan_command(nodes,'-e')
        for r in reports:
            port=r['port']
            print('Assigning',port,flush=True)
            kavlan_command(nodes,'-s','-i',vlan,'-m',port)
            status=kavlan_command(nodes,'-g','-m',port)
            if not status.strip().endswith(': '+str(vlan)):
                raise RuntimeError(f'Assignment did not stick: {port}: {status}')
        for i,r in enumerate(reports):
            h=r['host'];nic=r['isolated_iface']
            privileged(h,'ip link set dev '+shlex.quote(nic)+' up')
            try:ip=interface_ip(h,nic)
            except RuntimeError:
                privileged(h,'timeout 50 dhclient -1 -v '+shlex.quote(nic),timeout=75)
                ip=interface_ip(h,nic)
            route_guard(h,nic)
            mapped.append(dict(rank=i,host=h,ip=ip,iface=nic,gpu_class='allocated NVIDIA GPU',sat=f'SAT-DEMO-{i+1:03d}'))
        if len({x['ip'] for x in mapped})!=4:raise RuntimeError('Duplicate isolated IP')
        for n in mapped:
            for peer in mapped:
                if n==peer:continue
                route_guard(n['host'],n['iface'],peer['ip'],n['ip'])
                ssh(n['host'],'ping -I '+shlex.quote(n['iface'])+' -c 1 -W 2 '+shlex.quote(peer['ip']),timeout=12)
    else:
        print('SHARED BACKEND: no VLAN assignment, no changes to management NIC in setup.',flush=True)
        for i,h in enumerate(nodes):
            nic=default_interface(h)
            mapped.append(dict(rank=i,host=h,ip=interface_ip(h,nic),iface=nic,gpu_class='allocated NVIDIA GPU',sat=f'SAT-DEMO-{i+1:03d}'))
        if len({x['ip'] for x in mapped})!=4:raise RuntimeError('Duplicate management IPs')
        if len({x['iface'] for x in mapped})!=1:raise RuntimeError('Shared mode requires uniform management interface name for E2 launchers')
        validate_shared_routes(mapped)
    state=dict(oar_job_id=os.getenv('OAR_JOB_ID'),site=nodes[0].split('.')[1],
        vlan_id=int(vlan) if vlan is not None else None,network_mode=selected,nodes=mapped,created=time.time())
    (ROOT/'kavlan_state.json').write_text(json.dumps(state,indent=2))
    (ROOT/'network_state.json').write_text(json.dumps(state,indent=2))
    print('PASS:',selected,'configured:',[(n['host'],n['ip'],n['iface']) for n in mapped],flush=True)
    if selected=='selective_tc':
        print('WARNING: shared NIC tc modifies root qdisc during experiments; SSH exemptions are not a zero-risk guarantee.',flush=True)

if __name__=='__main__':main()
