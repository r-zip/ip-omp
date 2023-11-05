#!/bin/bash

JOBS=8
OUTPUT_DIR="./"

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "${OUTPUT_DIR}/results_small_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_gaussian \
    "${OUTPUT_DIR}/results_large_${SUFFIX}"

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "${OUTPUT_DIR}/results_small_${SUFFIX}_const"

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=$JOBS \
    --coeff-distribution=sparse_const \
    "${OUTPUT_DIR}/results_large_${SUFFIX}_const"