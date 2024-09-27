#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce \
    --datasets "sst2,agnews" \
    --methods "lora," \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
