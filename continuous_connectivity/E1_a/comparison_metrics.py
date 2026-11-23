#!/usr/bin/env python3
"""Cross-method comparison record, preserving distinctions between DDP, DiLoCo, gossip."""
import argparse,csv,json,statistics
from pathlib import Path

def readcsv(p):
    if not p.exists():return []
    with p.open(newline='') as f:return list(csv.DictReader(f))
def readjson(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
def num(v):
    try:return float(v)
    except (TypeError,ValueError):return None
def metric(rows,key,scale=1):
    v=[num(r.get(key)) for r in rows];v=[x*scale for x in v if x is not None]
    return dict(count=len(v),mean=round(statistics.mean(v),6) if v else None,
                median=round(statistics.median(v),6) if v else None,
                p95=round(sorted(v)[int(.95*(len(v)-1))],6) if v else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');a=p.parse_args();root=Path(a.run_dir)
    summary=readjson(root/'summary.json');manifest=readjson(root/'manifest.json')
    steps={p.stem:readcsv(p) for p in root.glob('steps_rank*.csv')}
    events=readcsv(root/'communication_events.csv')
    accuracy=readcsv(root/'validation_aggregate.csv')
    val=readcsv(root/'validation.csv')
    tcp=readcsv(root/'tcp_rank_pair_summary.csv')
    health=readjson(root/'socket_monitor_health.json')
    note=readjson(root/'tcp_measurement_notes.json')
    resources={p.stem:readcsv(p) for p in root.glob('resources_rank*.csv')}
    event_vol={}
    for r in events:
        k=r['rank'];event_vol[k]=event_vol.get(k,0)+int(r['payload_bytes_logical'])
    observed_tcp=sum(int(r.get('tcp_payload_observed_delta_bytes') or 0) for r in tcp)
    visibility=bool(health.get('directional_tcp_byte_intervals_observed'))
    result=dict(schema_version=1,experiment='E1_a',method='ddp',method_semantics='synchronous gradient AllReduce each step',
        world_size=summary.get('world_size'),global_batch_size=summary.get('global_batch_size'),
        completed_epochs=summary.get('completed_epochs'),steps_per_worker=summary.get('steps_per_worker'),
        global_samples_approx=summary.get('global_samples_approx'),wall_training_s=summary.get('wall_training_s'),
        best_rank0_validation_accuracy=summary.get('best_validation_accuracy'),
        target_first_s=summary.get('target_first_s'),
        final_rank0_validation_accuracy=num(val[-1].get('val_accuracy')) if val else None,
        final_mean_worker_validation_accuracy=num(accuracy[-1].get('mean_accuracy')) if accuracy else None,
        final_worker_accuracy_std=num(accuracy[-1].get('std_accuracy')) if accuracy else None,
        validation_scope='same holdout independently evaluated on each synchronized DDP replica',
        rank_step_ms={k:metric(v,'step_s',1000) for k,v in steps.items()},
        rank_gpu_util_pct={k:metric(v,'gpu_util_pct') for k,v in resources.items()},
        communication=dict(collective_events_logged=len(events),logical_bucket_bytes_by_rank=event_vol,
            scope='one gradient bucket submitted per worker, not directional physical traffic',
            coverage='traced bucket events only; sampling frequency configurable'),
        network=dict(tcp_payload_observed_delta_bytes=observed_tcp if visibility else None,
            per_peer_tcp_available=visibility,monitor_completeness_ok=health.get('monitor_completeness_ok'),
            scope='rootless sampled ss TCP_INFO; excludes missing/unsampled connection bytes and RDMA'),
        comparability_notes=['DDP backward includes overlapping gradient communication; E1_b local backward does not.',
            'DDP synchronized replicas are not independent gossip models.',
            'For performance comparison, match hardware, data split, batch, seed, runtime budget and transport.',
            'Logical bucket volume must not be compared as physical bytes to completed gossip sends.'])
    (root/'comparison_metrics.json').write_text(json.dumps(result,indent=2))
    print('Saved comparison_metrics.json')
if __name__=='__main__':main()
