#!/bin/bash -ex

source ./scripts/env.sh

metric=nce
mode="median"

overwrite=false
results_path=outputs/results/$model/$metric.jsonl
if [ -f $results_path ] && [ $overwrite = false ]; then
    echo "Results already computed. Skipping."
else
    mkdir -p $(dirname $results_path)
    python -m llmcal2.scripts.compute_results \
        --metric $metric \
        --root_results_dir outputs/adaptation/$model \
        --output_path $results_path \
        --encoder_results_dir outputs/adaptation/encoder_models
fi

bar_plots_dir=outputs/results/$model/bars
mkdir -p $bar_plots_dir
for dataset in ${DATASETS[@]}; do
    python -m llmcal2.scripts.plot_results \
        --dataset $dataset \
        --metric $metric \
        --results_path $results_path \
        --output_dir $bar_plots_dir \
        --mode $mode \
        --set_lim
done

# scatter_plots_dir=outputs/results/$model/scatters
# mkdir -p $scatter_plots_dir
# for dataset in ${DATASETS[@]}; do
#     python -m llmcal2.scripts.scatter_plots \
#         --dataset $dataset \
#         --root_results_dir outputs/adaptation/$model \
#         --output_dir $scatter_plots_dir
# done

