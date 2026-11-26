#!/usr/bin/env python3
"""Offline E3_1 normalized-metrics smoke test; no PyTorch dependency."""
import csv,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e3_analyze import metrics

def csvout(path,fields,values):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows(values)

class MetricTests(unittest.TestCase):
    def test_all_measurement_files_and_null_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trace=root/'trace.csv'
            fields=['src','dst','t_start_s','t_end_s','available','rate_bps','prop_delay_ms']
            seq=[]
            for i in range(4):
                for j in range(4):
                    if i==j:continue
                    for a,b,avail in ((0,20,1),(20,30,0),(30,40,1)):
                        seq.append(dict(src=f'SAT-DEMO-{i+1:03d}',dst=f'SAT-DEMO-{j+1:03d}',
                            t_start_s=a,t_end_s=b,available=avail,
                            rate_bps=1_000_000_000 if avail else 0,prop_delay_ms=2))
            csvout(trace,fields,seq)
            (root/'e2_run_config.json').write_text(json.dumps({
               'method':'gossip8','schedule_mode':'aware','init_unix_s':1000,
               'duration_s':40,'trace_file':str(trace),'trace_sha256':'test-sha'}))
            (root/'execution_status.json').write_text(json.dumps({'success':True,'failures':[]}))
            for i in range(4):
                name=root/f'e3_scheduler_decisions_rank{i}.csv'
                common=dict(rank=i,method='gossip8',comm_id=f'op{i}',scope='peer',
                            current_step=8,peer_rank=(i+1)%4,logical_bytes=128,
                            sender_model_version=8,message_created_unix=1008,
                            defer_duration_s=0,rendezvous_wait_s=0)
                row1=dict(common,scheduler_action='start',comm_ready_unix=1008,
                          comm_start_unix=1008,comm_complete_unix='')
                row2=dict(common,scheduler_action='complete',comm_ready_unix='',
                          comm_start_unix='',comm_complete_unix=1010,
                          communication_status='completed')
                csvout(name,list(dict.fromkeys([*row1,*row2])),[row1,row2])
                csvout(root/f'e2_events_rank{i}.csv',
                       ['rank','event','method','step','unix_s','detail'],
                       [dict(rank=i,event='model_apply',method='gossip8',step=10,
                             unix_s=1011,detail=f'gossip_from_rank={(i-1)%4};received_version=8'),
                        dict(rank=i,event='local_optimizer_step',method='gossip8',step=11,unix_s=1024)])
                csvout(root/f'steps_rank{i}.csv',['unix_time_s','step'],
                       [dict(unix_time_s=1024,step=11)])
                csvout(root/f'resources_rank{i}.csv',
                       ['unix_time_s','gpu_util_pct','status','tx_bytes_node','rx_bytes_node'],
                       [dict(unix_time_s=1022,gpu_util_pct=4,status='ok',tx_bytes_node=100,rx_bytes_node=200),
                        dict(unix_time_s=1032,gpu_util_pct=70,status='ok',tx_bytes_node=250,rx_bytes_node=300)])
            csvout(root/'validation_rank0.csv',['elapsed_s','val_accuracy'],
                   [dict(elapsed_s=5,val_accuracy=.48),dict(elapsed_s=12,val_accuracy=.55)])
            report=metrics(root)
            self.assertEqual(report['operation_completed_count'],4)
            self.assertEqual(report['successful_transmission_count'],4)
            self.assertEqual(report['completed_logical_bytes'],512)
            self.assertEqual(report['time_to_accuracy_s']['50'],12)
            self.assertIsNone(report['time_to_accuracy_s']['70'])
            self.assertIsNone(report['observed_tcp_bytes'])
            self.assertEqual(report['local_steps_during_union_outages'],4)
            for name in ('e3_communication_operations.csv','e3_recovery.csv',
                'e3_effective_H.csv','e3_outage_progress.csv','e3_message_staleness.csv',
                'e3_network_counters.csv','e3_gpu_outage_idle.csv','e3_summary.json'):
                self.assertTrue((root/name).is_file(),name)

if __name__=='__main__':unittest.main()
