#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=3
chmod +x scripts/experiments/nolog/*.sh

#./scripts/experiments/nomunolog/run_nb101.sh $experiment
./scripts/experiments/nolog/run_nb201.sh $experiment
./scripts/experiments/nolog/run_nb301.sh $experiment
./scripts/experiments/nolog/run_tnb101.sh $experiment
./scripts/experiments/nolog/run_tnb101_macro.sh $experiment