#!/bin/bash

SUFFIX='snr2'
JOBS=8

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "results_small_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "results_large_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "results_small_${SUFFIX}_const"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "results_large_${SUFFIX}_const"