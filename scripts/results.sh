#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce \
    --datasets "sst2,agnews,dbpedia,20newsgroups,banking77" \
    --methods no_adaptation,dp_calibration,lora \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
