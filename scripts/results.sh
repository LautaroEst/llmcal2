#!/bin/bash -ex

source ./scripts/env.sh

metric=nce

overwrite=true
results_path=outputs/results/$model/$metric.jsonl
if [ -f $results_path ] && [ $overwrite = false ]; then
    echo "Results already computed. Skipping."
else
    mkdir -p $(dirname $results_path)
    python -m llmcal2.scripts.compute_results \
        --metric $metric \
        --root_results_dir outputs/adaptation/$model \
        --output_path $results_path
fi

plots_dir=outputs/results/$model
mkdir -p $plots_dir
python -m llmcal2.scripts.plot_results \
    --metric $metric \
    --results_path $results_path \
    --output_dir $plots_dir
# # python -m llmcal2.scripts.plot_train_on_same_dataset \
# #     --psr $psr \
# #     --datasets $datasets \
# #     --methods $methods \
# #     --results_path $results_path \
# #     --output_dir $plots_dir
# python -m llmcal2.scripts.plot_crosstalk_dataset \
#     --psr $psr \
#     --datasets $datasets \
#     --methods $methods \
#     --results_path $results_path \
#     --output_dir $plots_dir




