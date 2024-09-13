#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=0
chmod +x scripts/bash_scripts/tmp_abl_batch2/*.sh
./scripts/bash_scripts/tmp_abl_batch2/run_nb101.sh $experiment
./scripts/bash_scripts/tmp_abl_batch2/run_nb201.sh $experiment
./scripts/bash_scripts/tmp_abl_batch2/run_nb301.sh $experiment
./scripts/bash_scripts/tmp_abl_batch2/run_tnb101.sh $experiment
./scripts/bash_scripts/tmp_abl_batch2/run_tnb101_macro.sh $experiment