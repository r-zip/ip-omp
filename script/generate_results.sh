#!/bin/bash

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=4 \
    --coeff-distribution=sparse_gaussian \
    results_small_fixed_iter

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=4 \
    --coeff-distribution=sparse_gaussian \
    results_large_fixed_iter

python -m ip_is_all_you_need.simulations \
    --setting=small \
    --jobs=4 \
    --coeff-distribution=sparse_const \
    results_small_fixed_iter_const

python -m ip_is_all_you_need.simulations \
    --setting=large \
    --jobs=4 \
    --coeff-distribution=sparse_const \
    results_large_fixed_iter_const