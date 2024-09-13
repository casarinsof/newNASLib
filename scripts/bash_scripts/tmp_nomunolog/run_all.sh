#!/bin/bash

experiment=$1
export CUDA_VISIBLE_DEVICES=0
chmod +x scripts/bash_scripts/tmp_nomunolog/*.sh

#./scripts/bash_scripts/tmp_nomunolog/run_nb101.sh $experiment
./scripts/bash_scripts/tmp_nomunolog/run_nb201.sh $experiment
./scripts/bash_scripts/tmp_nomunolog/run_nb301.sh $experiment
./scripts/bash_scripts/tmp_nomunolog/run_tnb101.sh $experiment
./scripts/bash_scripts/tmp_nomunolog/run_tnb101_macro.sh $experiment