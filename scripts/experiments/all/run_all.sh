#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=3
chmod +x scripts/experiments/all/*.sh

#./scripts/experiments/all/run_nb101.sh $experiment
#./scripts/experiments/all/run_nb201.sh $experiment
#./scripts/experiments/all/run_nb301.sh $experiment
./scripts/experiments/all/run_tnb101.sh $experiment
./scripts/experiments/all/run_tnb101_macro.sh $experiment