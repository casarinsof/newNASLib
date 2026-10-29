#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=0
chmod +x scripts/experiments/abl_batch2/*.sh
./scripts/experiments/abl_batch2/run_nb101.sh $experiment
./scripts/experiments/abl_batch2/run_nb201.sh $experiment
./scripts/experiments/abl_batch2/run_nb301.sh $experiment
./scripts/experiments/abl_batch2/run_tnb101.sh $experiment
./scripts/experiments/abl_batch2/run_tnb101_macro.sh $experiment