#! /bin/bash -ex
CUDA_VISIBLE_DEVICES=0
accelerator="gpu"
precision="bf16-true"
strategy="auto"
devices=1
num_nodes=1

batch_size=1
lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
optimizer="adamw"
weight_decay=0.0
max_steps=-1

model="llama3"
DATASETS="sst2 agnews dbpedia 20newsgroups banking77"

# Supported models
declare -A model2checkpoint=(
    ["tinyllama"]="$LIT_CHECKPOINTS/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    ["phi3"]="$LIT_CHECKPOINTS/microsoft/Phi-3-mini-4k-instruct"
    ["llama3"]="$LIT_CHECKPOINTS/meta-llama/Meta-Llama-3-8B"
)

# Number of classes in each dataset
declare -A dataset2numclasses=(
    ["sst2"]=2
    ["agnews"]=4
    ["dbpedia"]=14
    ["20newsgroups"]=20
    ["banking77"]=77
)

# Number of samples to use for adaptation
if [ $model == "tinyllama" ]; then
    declare -A dataset2samples=(
        ["sst2"]="8 32 512"
        ["agnews"]="4 16 256"
        ["dbpedia"]="2 8 128"
        ["20newsgroups"]="2 8 128"
        ["banking77"]="1 4 64"
    )
else
    declare -A dataset2samples=(
        ["sst2"]="8 512"
        ["agnews"]="4 256"
        ["dbpedia"]="2 128"
        ["20newsgroups"]="2 128"
        ["banking77"]="1 64"
    )
fi

# Seeds
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

# Hyperparameters
declare -A dataset2hparams=(
    ["sst2_8"]="--learning_rate=0.00001 --max_epochs=80 --val_check_interval=8 --accumulate_grad_batches=8"
    ["sst2_32"]="--learning_rate=0.0001 --max_epochs=50 --val_check_interval=16 --accumulate_grad_batches=16"
    ["sst2_512"]="--learning_rate=0.0001 --max_epochs=4 --val_check_interval=128 --accumulate_grad_batches=32"
    ["agnews_4"]="--learning_rate=0.00001 --max_epochs=80 --val_check_interval=8 --accumulate_grad_batches=8"
    ["agnews_16"]="--learning_rate=0.0001 --max_epochs=20 --val_check_interval=32 --accumulate_grad_batches=16"
    ["agnews_256"]="--learning_rate=0.0001 --max_epochs=4 --val_check_interval=128 --accumulate_grad_batches=32"
    ["dbpedia_2"]="--learning_rate=0.0001 --max_epochs=40 --val_check_interval=16 --accumulate_grad_batches=16"
    ["dbpedia_8"]="--learning_rate=0.0001 --max_epochs=20 --val_check_interval=32 --accumulate_grad_batches=16"
    ["dbpedia_128"]="--learning_rate=0.0001 --max_epochs=4 --val_check_interval=128 --accumulate_grad_batches=32"
    ["20newsgroups_2"]="--learning_rate=0.0001 --max_epochs=50 --val_check_interval=16 --accumulate_grad_batches=16"
    ["20newsgroups_8"]="--learning_rate=0.0001 --max_epochs=15 --val_check_interval=64 --accumulate_grad_batches=32"
    ["20newsgroups_128"]="--learning_rate=0.0001 --max_epochs=4 --val_check_interval=128 --accumulate_grad_batches=32"
    ["banking77_1"]="--learning_rate=0.0001 --max_epochs=20 --val_check_interval=32 --accumulate_grad_batches=16"
    ["banking77_4"]="--learning_rate=0.0001 --max_epochs=15 --val_check_interval=64 --accumulate_grad_batches=32"
    ["banking77_64"]="--learning_rate=0.0001 --max_epochs=3 --val_check_interval=128 --accumulate_grad_batches=32"
)

checkpoint=${model2checkpoint[$model]}
for dataset in $DATASETS; do
    output_dir="outputs/adaptation/$model/no_adaptation/$dataset/size=all/rs=all"
    for size in ${dataset2samples[$dataset]}; do
        for random_state in ${dataset2seed[$dataset"_"$size]}; do
            total_train_samples=$((size * dataset2numclasses[$dataset]))
            output_dir="outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state"
            if [ ! -f $output_dir/test_logits.csv ]; then
                mkdir -p $output_dir $output_dir/logs $output_dir/checkpoints
                # python -m llmcal2.scripts.lora \
                #     --data_dir outputs/prompts/generative/$dataset \
                #     --total_train_samples $total_train_samples \
                #     --val_prop 0.3 \
                #     --random_state $random_state \
                #     --checkpoint_dir $checkpoint \
                #     --batch_size $batch_size \
                #     --accelerator $accelerator \
                #     --strategy $strategy \
                #     --devices $devices \
                #     --num_nodes $num_nodes \
                #     --precision $precision \
                #     $lora_args \
                #     --optimizer=$optimizer \
                #     --weight_decay=$weight_decay \
                #     --max_steps=$max_steps \
                #     ${dataset2hparams[$dataset"_"$size]} \
                #     --output_dir $output_dir \
                #     --log_dir $output_dir/logs \
                #     --output_checkpoint_dir $output_dir/checkpoints
                # for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                #     cp $checkpoint/$file $output_dir/checkpoints
                # done
                lora_dir=$(ls ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/ | grep "lora_[0-9]*samples")
                python -m llmcal2.scripts.temp_copy_results \
                    --input_dir ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/predictions/train \
                    --output_dir $output_dir \
                    --split "train"
                python -m llmcal2.scripts.temp_copy_results \
                    --input_dir ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/predictions/validation \
                    --output_dir $output_dir \
                    --split "validation"
                python -m llmcal2.scripts.temp_copy_results \
                    --input_dir ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/predictions/test \
                    --output_dir $output_dir \
                    --split "test"
                for f in ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/checkpoint/*; do
                    if [ -f $f ]; then
                        cp -d $f $output_dir/checkpoints/
                    fi
                done                    
                cp ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/logs/events* $output_dir/logs/
                metrics_file=../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/logs/metrics.csv 
                if [ -f $metrics_file ]; then
                    cp $metrics_file $output_dir/logs/metrics.csv
                fi
                cp ../llmcal/experiments.llama3.10-09-2024/$dataset"_"$size"_"$random_state/basic_$dataset"_0-shot_litgpt"/lm_$model/$lora_dir/.cache/*.ckpt $output_dir/
            fi
        done
    done
done