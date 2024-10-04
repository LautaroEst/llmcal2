#! /bin/bash -ex

mkdir -p outputs/results
python -m llmcal2.scripts.plot_methods_vs_size \
    --model tinyllama \
    --metrics ner,nce,calloss_nce_trainontest,calloss_nce_xval \
    --datasets "sst2,20newsgroups" \
    --methods "lora,dp_calibration,temp_scaling,no_adaptation,lora_norm,lora_no_es,lora_norm_plus_temp_scaling_no_es,lora_plus_temp_scaling_no_es" \
    --results_dir outputs/adaptation \
    --output_dir outputs/results \
    --overwrite
