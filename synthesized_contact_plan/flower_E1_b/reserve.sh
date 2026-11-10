oarsub -I \
  -p "gpu_count >= 1" \
  -l "host=4/gpu=1,walltime=4:00:00" \
  -q "besteffort"


