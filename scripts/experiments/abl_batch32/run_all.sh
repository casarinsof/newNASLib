#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=1
chmod +x scripts/experiments/abl_batch32/*.sh
#./scripts/experiments/abl_batch32/run_nb101.sh $experiment
#./scripts/experiments/abl_batch32/run_nb201.sh $experiment
#./scripts/experiments/abl_batch32/run_nb301.sh $experiment
#./scripts/experiments/abl_batch32/run_tnb101.sh $experiment
./scripts/experiments/abl_batch32/run_tnb101_macro.sh $experiment