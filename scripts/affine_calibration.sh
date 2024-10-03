#! /bin/bash -ex
accelerator="cpu"
max_ls=40
learning_rate=5e-3
max_epochs=-1
methods="dp_calibration temp_scaling bias_only"

checkpoint=${model2checkpoint[$model]}
for size in ${dataset2samples[$dataset]}; do
    for random_state in ${dataset2seed[$dataset"_"$size]}; do
        for method in $methods; do
            total_train_samples=$((size * dataset2numclasses[$dataset]))
            output_dir="outputs/adaptation/$model/$method/$dataset/size=$size/rs=$random_state"
            if [ ! -f $output_dir/test_logits.csv ]; then
                mkdir -p $output_dir $output_dir/checkpoint $output_dir/logs
                python -m llmcal2.scripts.affine_calibration \
                    --train_logits outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state/train_logits.csv \
                    --train_labels outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state/train_label.csv \
                    --val_logits outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state/val_logits.csv \
                    --val_labels outputs/adaptation/$model/no_adaptation/$dataset/size=$size/rs=$random_state/val_label.csv \
                    --test_logits outputs/adaptation/$model/no_adaptation/$dataset/size=all/rs=all/test_logits.csv \
                    --test_labels outputs/adaptation/$model/no_adaptation/$dataset/size=all/rs=all/test_label.csv \
                    --val_prop 0.3 \
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