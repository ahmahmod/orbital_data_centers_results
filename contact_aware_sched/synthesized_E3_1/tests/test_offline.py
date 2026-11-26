import csv,json,os,pathlib,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from remote import site,profile,port_ref
from network_select import choose_network

class Tests(unittest.TestCase):
    def test_profiles(self):
        data=[('chuc-1.lille.grid5000.fr','ens15f1np1','eth1'),
              ('kinovis-1.grenoble.grid5000.fr','ens15f1np1','eth1'),
              ('gruss-1.nancy.grid5000.fr','eno33','eth2'),
              ('graffiti-5.nancy.grid5000.fr','ens4f0','eth2')]
        for h,nic,eth in data:
            self.assertEqual(profile(h)['iface'],nic)
            self.assertIn('-'+eth+'.',port_ref(h))
    def test_reject_unknown_cpu(self):
        with self.assertRaises(RuntimeError):profile('vercors2-2.grenoble.grid5000.fr')
    def test_network_selection_shared_fallback(self):
        from unittest.mock import patch
        hs=['graffiti-10.nancy.grid5000.fr']*4
        with patch('network_select.default_interface',return_value='ens4f0'):
            mode,reports=choose_network(hs,lambda port:'DEFAULT')
        self.assertEqual(mode,'selective_tc')

    def test_network_selection_isolated(self):
        from unittest.mock import patch
        hs=[f'chifflot-{i}.lille.grid5000.fr' for i in [2,5,7,8]]
        with patch('network_select.default_interface',return_value='ens5f0'), patch('network_select.route_guard',return_value=True):
            mode,reports=choose_network(hs,lambda port:port+': DEFAULT')
        self.assertEqual(mode,'kavlan')

    def test_selective_filters_preserve_ssh(self):
        s=(ROOT/'tc_agent.py').read_text()
        self.assertIn("'src_port','22'",s)
        self.assertIn("'dst_port','22'",s)
        self.assertIn("'prio',str(k+10)",s)

    def test_trace(self):
        rows=list(csv.DictReader((ROOT/'traces/directed_trace_4workers.csv').open()))
        self.assertEqual(len({(e['src'],e['dst']) for e in rows}),12)
        for edge in {(e['src'],e['dst']) for e in rows}:
            es=sorted((int(e['t_start_s']),int(e['t_end_s'])) for e in rows if (e['src'],e['dst'])==edge)
            self.assertEqual(es[0][0],0);self.assertEqual(es[-1][1],18000)
            self.assertTrue(all(a[1]==b[0] for a,b in zip(es,es[1:])))
            # The five cycles reproduce the original deterministic one-hour plan.
            one_hour=[(int(r['t_start_s']),int(r['t_end_s']),r['available'],r['rate_bps'],r['prop_delay_ms']) for r in rows if (r['src'],r['dst'])==edge and int(r['t_start_s'])<3600]
            for cycle in range(1,5):
                observed=[(int(r['t_start_s'])-cycle*3600,int(r['t_end_s'])-cycle*3600,r['available'],r['rate_bps'],r['prop_delay_ms']) for r in rows if (r['src'],r['dst'])==edge and cycle*3600<=int(r['t_start_s'])<(cycle+1)*3600]
                self.assertEqual(one_hour,observed)
        self.assertEqual(sum(int(e['available'])==0 for e in rows)>0,True)
    def test_runner_help(self):
        r=subprocess.run([sys.executable,str(ROOT/'runner.py'),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('gossip64',r.stdout)
    def test_analysis_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=pathlib.Path(tmp)
            (d/'e2_run_config.json').write_text(json.dumps(dict(init_unix_s=100)))
            p=subprocess.run([sys.executable,str(ROOT/'analyze.py'),str(d)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertTrue((d/'e2_analysis.json').exists())

if __name__=='__main__':unittest.main()

class ReservationTest(unittest.TestCase):
    def test_site_reservation_cli(self):
        with tempfile.TemporaryDirectory() as td:
            d=pathlib.Path(td)
            (d/'hostname').write_text('#!/bin/sh\necho f${SITE}\n')
            (d/'oarsub').write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            (d/'hostname').chmod(0o755);(d/'oarsub').chmod(0o755)
            for site,cluster,queue in [('lille','chifflot','abaca'),('grenoble','kinovis','abaca'),('nancy','gruss','abaca')]:
                e=os.environ.copy();e.update(SITE=site,PATH=str(d)+':'+e['PATH'])
                e.pop('OAR_JOB_ID',None)
                p=subprocess.run(['bash',str(ROOT/'reservation.sh')],env=e,text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stderr)
                self.assertIn("cluster='"+cluster+"'",p.stdout)
                self.assertIn('kavlan-local',p.stdout)
                self.assertIn('deploy',p.stdout)
                self.assertIn(queue,p.stdout)
