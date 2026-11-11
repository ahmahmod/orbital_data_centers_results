 oarsub -I \
  -p "host IN (graffiti-5, graffiti-7, graffiti-8, graffiti-9) AND gpu_count >= 1" \
  -l "host=4/gpu=1,walltime=4:00:00" \
  -q production

# oarsub -I \
#   -p "gpu_count >= 1" \
#   -l "host=4/gpu=1,walltime=4:00:00" \
#   -q "besteffort"