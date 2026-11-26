# Flower FedAvg — E3_1

Run fl8/fl64 with CASE=off or CASE=aware. Scheduling occurs at the fit callback return boundary, before Flower's gRPC upload. No client selection or change to local steps is applied. The first client is co-located with the Flower server; this must be reported as a confound. All-four coordination assumes the Grid'5000 shared filesystem is available to all ranks.
