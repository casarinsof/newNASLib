#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=2
chmod +x scripts/bash_scripts/tmp_my_zico/*.sh

#./scripts/bash_scripts/tmp_my_zico/run_nb101.sh $experiment
./scripts/bash_scripts/tmp_my_zico/run_nb201.sh $experiment
./scripts/bash_scripts/tmp_my_zico/run_nb301.sh $experiment
./scripts/bash_scripts/tmp_my_zico/run_tnb101.sh $experiment
./scripts/bash_scripts/tmp_my_zico/run_tnb101_macro.sh $experiment