#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=3
chmod +x scripts/experiments/abl_batch8/*.sh
./scripts/experiments/abl_batch8/run_nb101.sh $experiment
./scripts/experiments/abl_batch8/run_nb201.sh $experiment
./scripts/experiments/abl_batch8/run_nb301.sh $experiment
