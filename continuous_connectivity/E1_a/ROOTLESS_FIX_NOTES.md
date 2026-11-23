# E1_a rootless `ss` parser fix — 2026-11-07

## Verified failure
In run `E1_a_4hosts_1gpus_seed2026_20261107T115243407796615Z`, all four monitor CSVs report `connections_seen=0` despite `ss_returncode=0` and healthy NIC activity. The parser used fixed columns `cols[3]` and `cols[4]`, but `ss -H ... state established` may omit the ESTAB/state column. Consequently all TCP connections can be discarded without a warning.

## Changes
* Recognize the first two IPv4 `IP:port` tokens regardless of surrounding state/queue columns.
* Search ownership PID metadata in both the socket header and continuation.
* Collect extended `ss -e` diagnostics such as inode when present.
* Save a one-time `ss_preview_<host>.txt` to diagnose differing Grid5000 `ss` layouts.
* Add `connections_seen_total` to monitor health.
* Self-test uses the no-state `ss` output format that previously failed silently.

## Deployment
Copy these changed files into `synthesized_contact_plan/E1_a_new/`: `socket_counter_watch.py`, `socket_monitor_health.py`, `selftest_rootless.py`. Leave the other training and analysis files unchanged.

Run `python3 selftest_rootless.py` and then run a **new** experiment using the usual `./run.sh` command.

## Remaining limitations
If `ss -p` does not reveal PIDs on the reserved nodes, rank attribution can remain empty even when sockets are parsed. In that case inspect `connections_seen_total` and `ss_preview_<host>.txt`. This patch does not fabricate NCCL peer bytes from logical collective sizes or host-wide NIC counters.
