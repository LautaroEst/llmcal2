
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, log_softmax

from .compute_results import read_results
from .plot_results import parse_train_scenario as parse

def parse_train_scenario(ds):
    ds["result"] = None
    ds_new = parse(ds)
    ds_new = ds_new.drop("result")
    ds_new["logits"] = pd.read_csv(ds["logits"], index_col=0, header=None).values.astype(float)
    ds_new["labels"] = pd.read_csv(ds["labels"], index_col=0, header=None).values.flatten().astype(int)
    return ds_new

def logodds(log_p):
    return log_p - (np.log1p(-np.exp(-np.abs(log_p))) - np.maximum(0, log_p))
    

methods = OrderedDict([
    ("zero_shot", {"label": "Zero-shot", "color": "black", "hatch": None, "alpha": 1.0,}),
    ("instruct", {"label": "Instructions", "color": "tab:pink", "hatch": None,}),
    ("lora_fs-matched", {"label": "LoRA-FS (Matched)", "color": "tab:blue", "hatch": None,}),
    ("lora_fs-mismatched", {"label": "LoRA-FS (Mismatched)", "color": "tab:blue", "hatch": "/",}),
    ("lora_fs-all", {"label": "LoRA-FS (All)", "color": "tab:blue", "hatch": "x",}),
    ("lora_ans-matched", {"label": "LoRA-ANS (Matched)", "color": "tab:orange", "hatch": None,}),
    ("lora_ans-mismatched", {"label": "LoRA-ANS (Mismatched)", "color": "tab:orange", "hatch": "/",}),
    ("lora_ans-all", {"label": "LoRA-ANS (All)", "color": "tab:orange", "hatch": "x",}),
    # ("lora_norm-5-matched", {"label": "LoRA-Norm K=5 (Matched)", "color": "tab:green", "hatch": None,}),
    # ("lora_norm-5-mismatched", {"label": "LoRA-Norm K=5 (Mismatched)", "color": "tab:green", "hatch": "/",}),
    # ("lora_norm-5-all", {"label": "LoRA-Norm K=5 (All)", "color": "tab:green", "hatch": "x",}),
    # ("lora_norm-15-matched", {"label": "LoRA-Norm K=15 (Matched)", "color": "tab:brown", "hatch": None,}),
    # ("lora_norm-15-mismatched", {"label": "LoRA-Norm K=15 (Mismatched)", "color": "tab:brown", "hatch": "/",}),
    # ("lora_norm-15-all", {"label": "LoRA-Norm K=15 (All)", "color": "tab:brown", "hatch": "x",}),
    ("lora_fs-matched-no_es", {"label": "LoRA-FS (Matched, No ES)", "color": "blue", "hatch": None,}),
    # ("lora_fs-mismatched-no_es", {"label": "LoRA-FS (Mismatched, No ES)", "color": "blue", "hatch": "/",}),
    # ("lora_fs-all-no_es", {"label": "LoRA-FS (All, No ES)", "color": "blue", "hatch": "x",}),
    ("lora_ans-matched-no_es", {"label": "LoRA-ANS (Matched, No ES)", "color": "orange", "hatch": None,}),
    # ("lora_ans-mismatched-no_es", {"label": "LoRA-ANS (Mismatched, No ES)", "color": "orange", "hatch": "/",}),
    # ("lora_ans-all-no_es", {"label": "LoRA-ANS (All, No ES)", "color": "orange", "hatch": "x",}),
])

dataset2name = OrderedDict([
    ("sst2", {"name": "SST-2", "num_classes": 2}),
    ("agnews", {"name": "AG News", "num_classes": 4}),
    ("dbpedia", {"name": "DBpedia", "num_classes": 14}),
    ("20newsgroups", {"name": "20 Newsgroups", "num_classes": 20}),
    ("banking77", {"name": "Banking77", "num_classes": 77}),
])


def plot_scatter(data, dataset, output_path):
    fig, ax = plt.subplots(1, len(methods), figsize=(5*len(methods),5), sharex=False, sharey=True)
    for i, (method, kwargs) in enumerate(methods.items()):

        ax[i].set_xlabel("In-Task Log Odds")
        ax[i].set_title(f"{kwargs['label']}")
        ax[i].grid(True)

        group = data[(data["base_method"] == method) & (~data["is_calibrated"])]
        if len(group) == 0:
            continue
        in_task_logprobs = logsumexp(np.vstack(group["logits"].values), axis=1)
        in_task_logodds = logodds(in_task_logprobs)
        ground_truth_logprob = np.hstack([log_softmax(logits, axis=1)[np.arange(len(labels)),labels] for logits, labels in zip(group["logits"], group["labels"])])
        ground_truth_logodds = logodds(ground_truth_logprob)
        ax[i].scatter(
            in_task_logodds, 
            ground_truth_logodds, 
            label=kwargs["label"], 
            alpha=0.7,
            color="tab:blue",
        )

        if method == "zero_shot":
            group = data[(data["base_method"] == "dp_calibration")]
        else:
            group = data[(data["base_method"] == method) & data["is_calibrated"]]

        if len(group) == 0:
            continue

        in_task_logprobs = logsumexp(np.vstack(group["logits"].values), axis=1)
        in_task_logodds = logodds(in_task_logprobs)
        ground_truth_logprob = np.hstack([log_softmax(logits, axis=1)[np.arange(len(labels)),labels] for logits, labels in zip(group["logits"], group["labels"])])
        ground_truth_logodds = logodds(ground_truth_logprob)
        ax[i].scatter(
            in_task_logodds, 
            ground_truth_logodds, 
            alpha=0.3,
            color="tab:orange",
        )

    fig.suptitle(f"{dataset2name[dataset]['name']}")
    ax[0].set_ylabel("Ground Truth Log Odds")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)



def main(
    dataset,
    root_results_dir,
    output_dir,
):
    root_results_dir = Path(root_results_dir)
    output_dir = Path(output_dir)
    data = read_results(root_results_dir)
    data = data[data["test_dataset"] == dataset]
    data = data[data["test_lst"].str.startswith("test_")]
    if len(data) == 0:
        return
    data = data.apply(parse_train_scenario, axis=1)
    plot_scatter(data, dataset, output_dir / f"{dataset}.png")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)