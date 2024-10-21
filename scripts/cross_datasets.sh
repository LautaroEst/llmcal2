#! /bin/bash -ex
accelerator="gpu"
precision="bf16-true"
strategy="auto"
devices=1
num_nodes=1

src_size=2
src_rs=435
src_dataset="dbpedia"
tgt_dataset="sst2"

checkpoint=${model2checkpoint[$model]}
output_dir="outputs/adaptation/${model}_${src_dataset}/no_adaptation/$tgt_dataset/size=all/rs=all"
test_list=${dataset2testlist[$dataset]}
if [ ! -f $output_dir/test_logits.csv ]; then
    mkdir -p $output_dir
    python -m llmcal2.scripts.predict_trained_model \
        --data_dir outputs/prompts/generative/$dataset \
        --checkpoint_dir $checkpoint \
        --test_list lists/$dataset/$test_list.txt \
        --batch_size 1 \
        --accelerator $accelerator \
        --strategy $strategy \
        --devices $devices \
        --num_nodes $num_nodes \
        --precision $precision \
        --output_dir $output_dir
fi