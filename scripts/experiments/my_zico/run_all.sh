#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=2
chmod +x scripts/experiments/my_zico/*.sh

#./scripts/experiments/my_zico/run_nb101.sh $experiment
./scripts/experiments/my_zico/run_nb201.sh $experiment
./scripts/experiments/my_zico/run_nb301.sh $experiment
./scripts/experiments/my_zico/run_tnb101.sh $experiment
./scripts/experiments/my_zico/run_tnb101_macro.sh $experiment