predictors=(swap fisher grad_norm grasp jacov snip synflow params flops l2_norm plain zen zico nwot)

start_seed=$1
if [ -z "$start_seed" ]
then
    start_seed=0
fi

# folders:
base_file=scripts
s3_folder=nb301
out_dir=$s3_folder\_$start_seed
config_root=configs
experiment=ablation_b2

# search space / data:
search_space=nasbench301
dataset=cifar10

# other variables:
trials=1
end_seed=$(($start_seed + $trials - 1))
test_size=1000
batch_size=2
# create config files
for predictor in ${predictors[@]}
do
    python $base_file/create_configs.py --predictor $predictor \
    --test_size $test_size --start_seed $start_seed --trials $trials --out_dir $out_dir \
    --dataset=$dataset --search_space $search_space --experiment $experiment --config_root=$config_root --batch_size $batch_size
done

# run experiments
for t in $(seq $start_seed $end_seed)
do
    for predictor in ${predictors[@]}
    do
        config_file=${config_root}/${experiment}/${predictor}/${search_space}-${start_seed}/${dataset}/config_${start_seed}.yaml
        echo $config_file
        echo ================running $predictor trial: $t for cifar10 =====================
        python naslib/runners/runner.py --config-file $config_file
    done
done
