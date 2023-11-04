#!/bin/bash

for snr in 5 10 15 20
do
    python -m ip_omp.figures \
        --max-m-large=120 \
        --together \
        --snr=$snr \
        --metric=nmse_x_mean \
        --semilogy \
        rebuttal_final/small.parquet \
        rebuttal_final/large.parquet \
        "results_snr_${snr}"
done