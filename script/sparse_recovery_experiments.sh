#!/bin/bash

if [[ $1 == "--test" ]] ; then
    JOBS=1
    EXTRA_FLAGS="--test"
    EXTRA_SUFFIX="_test"
else
    JOBS=8
    EXTRA_FLAGS=""
    EXTRA_SUFFIX=""
fi

RESULTS_DIR="./results"

mkdir -p $RESULTS_DIR

for problem_size in small large
do
    for coeff_dist in gaussian const
    do
        python -m ip_omp.simulations \
            $EXTRA_FLAGS \
            --jobs=$JOBS \
            --noise-setting=noiseless \
            --problem-size=$problem_size \
            --coeff-distribution="sparse_${coeff_dist}" \
            "${RESULTS_DIR}/results_noiseless_${problem_size}_${coeff_dist}${EXTRA_SUFFIX}"
    done
done

for problem_size in small large
do
    python -m ip_omp.simulations \
        $EXTRA_FLAGS \
        --jobs=$JOBS \
        --noise-setting=noisy \
        --problem-size=$problem_size \
        --coeff-distribution=sparse_gaussian \
        "${RESULTS_DIR}/results_noisy_${problem_size}${EXTRA_SUFFIX}"
done