# E2 network backends (four physical GPU hosts)

## Selection

`E2_CONFIRM_NETWORK=1 ./setup.sh` first deploys the four nodes, then runs a **read-only selection**:

- `kavlan` if each node has a separately routed, recognized KaVLAN experimental port. Then it sets membership, runs DHCP and checks all 12 links.
- `selective_tc` otherwise. No VLAN ports or IP addresses are changed. E2 uses the four management IPv4 addresses for peer-to-peer training traffic. All four workers use the same selected backend.

`NETWORK_MODE=kavlan` forces safe isolated NICs and **fails**, rather than falling back.
`NETWORK_MODE=selective_tc` forces selective shaping with no VLAN assignment.

Read the selection without applying it: `python3 bootstrap.py` (after deployment).

## Filtered tc behavior

Default HTB class `1:999`: no satellite delay/loss. Priority 1/2 flower filters explicitly route TCP **source or destination port 22** to the default class. Lower-priority per-destination-worker-IP filters attach netem classes with synthetic bandwidth, delay and outage periods. This covers dynamic NCCL/DDP communication ports, and Flower/Gossip/DiLoCo peer traffic. Only egress traffic is shaped; the reverse worker link is handled on the reverse source node.

**Caution:** Shaping peer-IP traffic can affect unrelated traffic between allocated hosts (NFS, SSH tunnels, diagnostics); only port 22 is explicitly bypassed. HTB root installation can still disrupt management due to kernel configuration, pre-existing qdiscs, shared link congestion or misclassification. SSH bypass is not a full safety guarantee. Preserve an independent administrative recovery path, do a short dry run, and monitor the node. DNS, storage and Grid'5000 infrastructure traffic is not targeted unless its destination is another worker IP.

The script does not erase existing E2 qdiscs silently; use `./cleanup.sh` when needed. Do not run multiple E2 tc experiments concurrently on the same hosts. The deployed image must have NVIDIA drivers + supported PyTorch.

## Commands

```bash
# In a four-GPU deploy reservation on its site frontend:
E2_CONFIRM_NETWORK=1 ./setup.sh
cat network_state.json

# Pilot: no qdisc changes
DATASET=synthetic METHOD=gossip8 MAX_SECONDS=180 ./run.sh

# Real outage replay on shared management NIC; two explicit confirmations
TC_APPLY=1 E2_CONFIRM_SHAPING=1 E2_CONFIRM_SHARED_TC=1 \
  DATASET=synthetic METHOD=gossip8 MAX_SECONDS=180 ./run.sh

# Read-only qdisc inventory; use E2_CONFIRM_CLEANUP=1 to remove E2 qdiscs
./cleanup.sh
```

Existing 7 methods preserved: `ddp`, `gossip8`, `gossip64`, `diloco8`, `diloco64`, `fl8`, `fl64`.

**Reservation:** In shared mode, KaVLAN is not needed for replay. The current reservation script retains a KaVLAN-local resource to allow auto-selection of isolated mode; it remains a deploy job. On a site without available KaVLAN, future revisions can provide an optional non-KaVLAN deploy reservation.
