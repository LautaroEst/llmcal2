#! /bin/bash -ex
CUDA_VISIBLE_DEVICES=0
accelerator="cpu"
max_ls=40
learning_rate=1e-2
max_epochs=30

declare -A model2checkpoint=(
    ["tinyllama"]="$LIT_CHECKPOINTS/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # ["phi3"]="$LIT_CHECKPOINTS/microsoft/Phi-3-mini-4k-instruct"
    # ["llama3"]="$LIT_CHECKPOINTS/meta-llama/Meta-Llama-3-8B"
)

DATASETS="sst2 agnews dbpedia 20newsgroups banking77"
methods="dp_calibration temp_scaling bias_only"

declare -A dataset2numclasses=(
    ["sst2"]=2
    ["agnews"]=4
    ["dbpedia"]=14
    ["20newsgroups"]=20
    ["banking77"]=77
)

declare -A dataset2samples=(
    ["sst2"]="8 32 512"
    ["agnews"]="4 16 256"
    ["dbpedia"]="2 8 128"
    ["20newsgroups"]="2 8 128"
    ["banking77"]="1 4 64"
)

declare -A dataset2seed=(
    ["sst2_8"]="639 923 932 6391 9322"
    ["sst2_32"]="1564 1738 1783 15641 17832"
    ["sst2_512"]="111 121 767 890 999"
    ["agnews_4"]="295 926 962 2951 9622"
    ["agnews_16"]="738 564 783 5641 7832"
    ["agnews_256"]="493 821 812 4931 8212"
    ["dbpedia_2"]="435 927 972 4351 9722"
    ["dbpedia_8"]="338 364 383 3641 3832"
    ["dbpedia_128"]="129 131 543 878 909"
    ["20newsgroups_2"]="435 927 972 4351 9722"
    ["20newsgroups_8"]="338 364 383 3641 3832"
    ["20newsgroups_128"]="129 131 543 878 909"
    ["banking77_1"]="322 444 848 858 868"
    ["banking77_4"]="295 926 962 2951 9622"
    ["banking77_64"]="131 888 893 912 933"
)

for model in ${!model2checkpoint[@]}; do
    checkpoint=${model2checkpoint[$model]}
    for dataset in $DATASETS; do
        for size in ${dataset2samples[$dataset]}; do
            for random_state in ${dataset2seed[$dataset"_"$size]}; do
                for method in $methods; do
                    total_train_samples=$((size * dataset2numclasses[$dataset]))
                    output_dir="outputs/adaptation/$model/lora_plus_${method}_no_es/$dataset/size=$size/rs=$random_state"
                    if [ ! -f $output_dir/train_logits.csv ]; then
                        mkdir -p $output_dir $output_dir/checkpoint $output_dir/logs
                        python -m llmcal2.scripts.affine_calibration \
                            --train_logits outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state/val_logits.csv \
                            --train_labels outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state/val_label.csv \
                            --test_logits outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state/test_logits.csv \
                            --test_labels outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state/test_label.csv \
                            --val_prop 0 \
                            --random_state $random_state \
                            --method $method \
                            --max_ls $max_ls \
                            --learning_rate $learning_rate \
                            --max_epochs $max_epochs \
                            --accelerator $accelerator \
                            --output_dir $output_dir \
                            --checkpoint_dir $output_dir/checkpoint \
                            --log_dir $output_dir/logs
                    fi
                done
            done
        done
    done
done