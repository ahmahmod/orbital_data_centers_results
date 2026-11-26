#!/usr/bin/env python3
"""Read-only selection of isolated KaVLAN or selective shared-NIC tc for four GPU hosts.

Never alter a management NIC or VLAN in this module.
"""
import ipaddress,re,shlex
from remote import ssh,profile,port_ref,route_guard


def default_interface(host):
    routes=ssh(host,'ip -4 route show default')
    names=re.findall(r'\bdev\s+(\S+)',routes)
    if not names or len(set(names))!=1:
        raise RuntimeError(f'{host}: unable to identify one management/default interface: {routes}')
    return names[0]


def interface_ip(host,iface):
    out=ssh(host,'ip -4 -o addr show dev '+shlex.quote(iface)+' scope global')
    ips=re.findall(r'\binet\s+([0-9.]+)/\d+',out)
    if len(ips)!=1:raise RuntimeError(f'{host}: expected exactly one global IPv4 on {iface}, got {ips}')
    ipaddress.IPv4Address(ips[0]);return ips[0]


def choose_network(nodes,lookup_port):
    """Prefer isolated NIC only if every node has safe NIC and recognized KaVLAN port.
    Conservative shared fallback across all four nodes; no VLAN writes here.
    """
    reports=[];isolated=True
    for host in nodes:
        mgmt=default_interface(host)
        try:
            p=profile(host)
            if p.get('isolated_available') is False:
                raise RuntimeError('cluster has no usable isolated experimental NIC; using guarded selective_tc')
            nic=p['iface']
            if nic==mgmt:raise RuntimeError('candidate NIC is management default-route NIC')
            route_guard(host,nic)
            port=port_ref(host)
            status=lookup_port(port)
            if 'unknown_node' in status.lower():raise RuntimeError('KaVLAN port unrecognized: '+port)
            reports.append(dict(host=host,management_iface=mgmt,isolated_iface=nic,port=port,port_status=status,choice='isolated'))
        except Exception as exc:
            isolated=False
            reports.append(dict(host=host,management_iface=mgmt,choice='shared',reason=str(exc)))
    if isolated and len({r['isolated_iface'] for r in reports})!=1:
        isolated=False
        for r in reports:r['choice']='shared';r['reason']='worker NIC names differ; requires uniform NIC for launcher'
    if isolated:
        return 'kavlan',reports
    for r in reports:r['choice']='shared'
    return 'selective_tc',reports


def validate_shared_routes(nodes):
    for n in nodes:
        host=n['host'];nic=n['iface']
        if nic!=default_interface(host):raise RuntimeError(f'{host}: shared iface must equal management/default interface')
        current=interface_ip(host,nic)
        if n['ip']!=current:raise RuntimeError(f'{host}: unexpected IP change {current} != {n["ip"]}')
        for p in nodes:
            if p['host']==host:continue
            rt=ssh(host,'ip -4 route get '+shlex.quote(p['ip']))
            if not re.search(r'\bdev\s+'+re.escape(nic)+r'\b',rt):
                raise RuntimeError(f'{host}->{p["host"]}: wrong shared interface: {rt}')
