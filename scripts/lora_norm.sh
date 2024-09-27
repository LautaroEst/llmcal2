#! /bin/bash -ex
accelerator="gpu"
precision="bf16-true"
strategy="auto"
devices=1
num_nodes=1

batch_size=1
accumulate_grad_batches=8
max_epochs=-1
val_check_interval=16
learning_rate=0.0001
lora_args="--lora_r=8 --lora_alpha=16 --lora_dropout=0.05 --lora_query --lora_key --lora_value --lora_projection --lora_mlp --lora_head"
optimizer="adamw"
weight_decay=0.0
max_steps=-1

checkpoint=${model2checkpoint[$model]}
for dataset in $DATASETS; do
    for size in ${dataset2samples[$dataset]}; do
        for random_state in ${dataset2seed[$dataset"_"$size]}; do
            total_train_samples=$((size * dataset2numclasses[$dataset]))
            use_train_samples_as_val=-1
            output_dir="outputs/adaptation/$model/lora/$dataset/size=$size/rs=$random_state"
            if [ ! -f $output_dir/test_logits.csv ]; then
                mkdir -p $output_dir $output_dir/logs $output_dir/checkpoints
                python -m llmcal2.scripts.lora \
                    --data_dir outputs/prompts/generative/$dataset \
                    --total_train_samples $total_train_samples \
                    --val_prop 0.3 \
                    --use_train_samples_as_val $use_train_samples_as_val \
                    --random_state $random_state \
                    --checkpoint_dir $checkpoint \
                    --norm \
                    --batch_size $batch_size \
                    --accelerator $accelerator \
                    --strategy $strategy \
                    --devices $devices \
                    --num_nodes $num_nodes \
                    --precision $precision \
                    --max_epochs=$max_epochs \
                    --max_steps=$max_steps \
                    --val_check_interval=$val_check_interval \
                    --accumulate_grad_batches=$accumulate_grad_batches \
                    $lora_args \
                    --optimizer=$optimizer \
                    --learning_rate=$learning_rate \
                    --weight_decay=$weight_decay \
                    --output_dir $output_dir \
                    --log_dir $output_dir/logs \
                    --output_checkpoint_dir $output_dir/checkpoints
                for file in config.json generation_config.json model_config.yaml tokenizer.json tokenizer.model tokenizer_config.json; do
                    cp $checkpoint/$file $output_dir/checkpoints
                done
            fi
        done
    done
done