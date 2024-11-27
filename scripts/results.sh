#!/bin/bash -ex

source ./scripts/env.sh

metric=nce
mode="median"

overwrite=true
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

# Debugging bar plots:
bar_plots_dir=outputs/results/$model/bars
mkdir -p $bar_plots_dir
for dataset in ${DATASETS[@]}; do
    python -m llmcal2.scripts.plot_results \
        --dataset $dataset \
        --metric $metric \
        --num_samples ${dataset2trainsize[$dataset]} \
        --results_path $results_path \
        --output_dir $bar_plots_dir \
        --mode $mode \
        --set_lim
done

# Results matched:
matched_plots_dir=outputs/results/$model/matched
mkdir -p $matched_plots_dir
python -m llmcal2.scripts.results_matched \
    --methods "no_adaptation lora_fs lora_ans" \
    --cal_methods "_plus_dp_cal " \
    --datasets "${DATASETS[*]}" \
    --metric $metric \
    --results_path $results_path \
    --outputs_dir $matched_plots_dir
# pdflatex -output-directory $matched_plots_dir $matched_plots_dir/matched.tex

# Training samples:
samples_plots_dir=outputs/results/$model/samples
mkdir -p $samples_plots_dir
python -m llmcal2.scripts.plot_metric_vs_samples \
    --datasets "${DATASETS[*]}" \
    --metric $metric \
    --methods "no_adaptation no_adaptation_plus_dp_cal" \
    --results_path $results_path \
    --output_dir $samples_plots_dir


# scatter_plots_dir=outputs/results/$model/scatters
# mkdir -p $scatter_plots_dir
# for dataset in ${DATASETS[@]}; do
#     python -m llmcal2.scripts.scatter_plots \
#         --dataset $dataset \
#         --root_results_dir outputs/adaptation/$model \
#         --output_dir $scatter_plots_dir
# done

