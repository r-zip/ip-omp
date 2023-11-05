#!/bin/bash

JOBS=8
RESULTS_DIR="./results"

mkdir -p $RESULTS_DIR

for problem_size in small large
do
    for coeff_dist in gaussian const
    do
        python -m ip_omp.simulations \
            --jobs=$JOBS \
            --noise-setting=noiseless \
            --problem-size=$problem_size \
            --coeff-distribution="sparse_${coeff_dist}" \
            "${RESULTS_DIR}/results_noiseless_${problem_size}_${coeff_dist}"
    done
done

for problem_size in small large
do
    python -m ip_omp.simulations \
        --jobs=$JOBS \
        --noise-setting=noisy \
        --problem-size=$problem_size \
        --coeff-distribution=sparse_gaussian \
        "${RESULTS_DIR}/results_noisy_${problem_size}"
done