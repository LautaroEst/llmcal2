#!/bin/bash -ex

source ./scripts/env.sh

psr=nce_xval
datasets=sst2,agnews
methods=no_adaptation,lora_fs,lora_ans,lora_norm-5

overwrite=true
results_path=outputs/results/$model/results.csv
if [ -f $results_path ] && [ $overwrite = false ]; then
    echo "Results already computed. Skipping."
else
    mkdir -p $(dirname $results_path)
    python -m llmcal2.scripts.compute_results \
        --psr $psr \
        --datasets $datasets \
        --methods $methods \
        --root_results_dir outputs/adaptation/$model \
        --output_path $results_path
fi

plots_dir=outputs/results/$model
mkdir -p $plots_dir
# python -m llmcal2.scripts.plot_train_on_same_dataset \
#     --psr $psr \
#     --datasets $datasets \
#     --methods $methods \
#     --results_path $results_path \
#     --output_dir $plots_dir
python -m llmcal2.scripts.plot_crosstalk_dataset \
    --psr $psr \
    --datasets $datasets \
    --methods $methods \
    --results_path $results_path \
    --output_dir $plots_dir




