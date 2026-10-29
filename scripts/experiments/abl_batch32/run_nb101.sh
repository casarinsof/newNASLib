predictors=(swap fisher grad_norm grasp jacov snip synflow params flops l2_norm plain zen zico nwot)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=scripts
s3_folder=nb101
out_dir=$s3_folder\_$start_seed
config_root=configs/ablation
experiment=ablation_b32

# search space / datasets:
search_space=nasbench101
datasets=(cifar10)

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=1000
batch_size=32


# create config files
for dataset in ${datasets[@]}
do
 for predictor in ${predictors[@]}
 do
    python3 $base_file/create_configs.py --predictor $predictor \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --batch_size $batch_size --experiment $experiment --config_root=$config_root
 done
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for dataset in ${datasets[@]}
    do
       for predictor in ${predictors[@]}
       do
          config_file=${config_root}/${experiment}/${predictor}/${search_space}-${start_seed}/${dataset}/config_${start_seed}.yaml
          echo $config_file
        echo ================running $predictor trial: $t for $dataset =====================
        python3 naslib/runners/runner.py --config-file $config_file
       done
    done
done
