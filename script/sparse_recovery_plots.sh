#!/bin/bash

RESULTS_DIR="./results"
OUTPUT_DIR="./plots"
FORMAT="png"

mkdir -p $OUTPUT_DIR

for coeff_dist in gaussian const
do
    python -m ip_omp.figures plot-noiseless \
        --coeff-distribution="sparse_${coeff_dist}" \
        --save-dir=$OUTPUT_DIR \
        --save-file-format=$FORMAT \
        "${RESULTS_DIR}/results_noiseless_small_${coeff_dist}/results.parquet" \
        "${RESULTS_DIR}/results_noiseless_large_${coeff_dist}/results.parquet"
done

python -m ip_omp.figures plot-noisy \
    --save-dir=$OUTPUT_DIR \
    --save-file-format=$FORMAT \
    "${RESULTS_DIR}/results_noisy_small/results.parquet" \
    "${RESULTS_DIR}/results_noisy_large/results.parquet"

