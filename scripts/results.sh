#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce,calloss_nce_xval \
    --datasets "sst2,dbpedia" \
    --methods "lora,dp_calibration,temp_scaling,no_adaptation,lora_norm,lora_normdp" \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
