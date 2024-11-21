#!/bin/bash -ex

source ./scripts/env.sh

val_prop=0.3
for dataset in "${DATASETS[@]}"; do
    list_dir=lists/$dataset
    data_dir=data/$dataset
    total_train_size=${dataset2trainsize[$dataset]}
    test_size=${dataset2testsize[$dataset]}

    mkdir -p $list_dir $data_dir
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
        --total_train_size $total_train_size --val_prop $val_prop --repetitions $num_seeds --seed $base_seed
    python -m llmcal2.scripts.create_lists --dataset $dataset --lists_dir $list_dir --data_dir $data_dir \
        --test_size $test_size --seed $base_seed

    for num_seed in $(seq 0 $((num_seeds-1))); do
        cat $list_dir/train_${dataset2trainsize[$dataset]}_0.3_$seed.txt $list_dir/val_${dataset2trainsize[$dataset]}_0.3_$seed.txt > $list_dir/train_${dataset2trainsize[$dataset]}_0.0_$seed.txt
        for n_shots in 2 4 8 16 32; do
            shots_list=${n_shots}shots_${num_seed}
            head -n $n_shots $list_dir/val_${dataset2trainsize[$dataset]}_${val_prop}_${num_seed}.txt > $list_dir/$shots_list.txt
        done
    done
done

