predictors=(swap) #(fisher grad_norm grasp jacov snip synflow params flops)


start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=scripts
s3_folder=tnb101_macro_reg10
config_root=configs
experiment=benchmarks
out_dir=$s3_folder\_$start_seed

# search space / data:
searchspaces=(transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=1000
batch_size=32
# create config files
for searchspace in "${searchspaces[@]}"
do
  for dataset in ${datasets[@]}
  do
   for i in $(seq 0 $((${#predictors[@]}-1)) )
   do
      predictor=${predictors[$i]}
      python $base_file/create_configs.py --predictor $predictor --test_size $test_size \
        --start_seed $start_seed --trials $trials --out_dir $out_dir --dataset=$dataset \
        --search_space $searchspace --batch_size $batch_size --experiment benchmarks --config_root=$config_root
   done
  done
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
for searchspace in "${searchspaces[@]}"
do
   for dataset in ${datasets[@]}
   do
      for predictor in ${predictors[@]}
      do
          config_file=${config_root}/${experiment}/${predictor}/${searchspace}-${start_seed}/${dataset}/config_${start_seed}.yaml
          echo $config_file
          echo ================running $predictor trial: $t for $dataset =====================
          python naslib/runners/runner.py --config-file $config_file
      done
   done
  done
done

