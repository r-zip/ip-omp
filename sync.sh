#!/bin/bash

rsync -av \
      --exclude .git \
      --exclude data \
      --exclude .ipynb_checkpoints \
      --exclude .pytest_cache \
      --exclude venv \
      --exclude *.egg-info \
      --exclude __pycache__ \
      --exclude results \
      . \
      cis:~/ip_omp
