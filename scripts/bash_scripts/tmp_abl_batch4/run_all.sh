#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=1
chmod +x scripts/bash_scripts/tmp_abl_batch4/*.sh
./scripts/bash_scripts/tmp_abl_batch4/run_nb101.sh $experiment
./scripts/bash_scripts/tmp_abl_batch4/run_nb201.sh $experiment
./scripts/bash_scripts/tmp_abl_batch4/run_nb301.sh $experiment
./scripts/bash_scripts/tmp_abl_batch4/run_tnb101.sh $experiment
./scripts/bash_scripts/tmp_abl_batch4/run_tnb101_macro.sh $experiment