# Changes from prior E2 standalone release

- Lille default reservation now `-t deploy`, `abaca`, Chifflot and KaVLAN-local; same-site GPU/cluster resource checks.
- `setup.sh` deploys four GPU nodes with `kadeploy3 -k` (as `emulated_dist_ai/cont_run.sh` did) before network inspection.
- All setup, training, and cleanup use deployed `ssh root@host`, not `oarsh`.
- KaVLAN commands run on the frontend or via SSH to the frontend with explicit OAR job ID.
- Setup validates data NIC, DHCP, 12 peer routes, and deployed GPU runtime. It does not install NVIDIA drivers or promise CUDA compatibility on the default Debian image.
- `run.sh` keeps training separate from deployment so nodes are not wiped before each run.
