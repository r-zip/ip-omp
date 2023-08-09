#!/bin/bash

for snr in 5 10 15 20
do
    python -m ip_is_all_you_need.figures \
        --max-m-large=120 \
        --together \
        --snr=$snr \
        --metric=nmse_x_mean \
        --semilogy \
        all_results_snr/small.parquet \
        all_results_snr/large.parquet \
        "results_snr_${snr}"
done