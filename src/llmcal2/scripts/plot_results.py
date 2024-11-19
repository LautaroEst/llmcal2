
from typing import OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

ENCODER_MODELS = [
    "distilbert-base-uncased",
    # "deberta-v2-xlarge",
    "roberta-large-mnli",
]

def parse_train_scenario(ds):
    if ds["method"] in ENCODER_MODELS:
        base_method = ds["method"]
        is_calibrated = False
    elif ds["method"] == "no_adaptation" and ds["train_lists"] == ["all"]:
        base_method = "zero_shot"
        is_calibrated = False
    elif ds["method"] == "instruct" and ds["train_lists"] == ["all"]:
        base_method = "instruct"
        is_calibrated = False
    elif ds["method"] == "no_adaptation_plus_dp_cal":
        base_method = "dp_calibration"
        is_calibrated = True
    elif ds["method"] == "instruct_plus_dp_cal":
        base_method = "instruct"
        is_calibrated = True
    elif ds["method"].endswith("_plus_dp_cal"):
        if len(ds["train_lists"]) == 1 and ds["train_lists"][0].split("_")[0] == ds["test_dataset"]:
            base_method = ds["method"].split("_plus_dp_cal")[0] + "-matched"
            is_calibrated = True
        elif len(ds["train_lists"]) == 4 and ds["test_dataset"] not in [lst.split("_")[0] for lst in ds["train_lists"]]:
            base_method = ds["method"].split("_plus_dp_cal")[0] + "-mismatched"
            is_calibrated = True
        elif len(ds["train_lists"]) == 5:
            base_method = ds["method"].split("_plus_dp_cal")[0] + "-all"
            is_calibrated = True
    elif ds["method"].endswith("_no_es"):
        if len(ds["train_lists"]) == 1 and ds["train_lists"][0].split("_")[0] == ds["test_dataset"]:
            base_method = ds["method"].split("_no_es")[0] + "-matched-no_es"
            is_calibrated = False
        elif len(ds["train_lists"]) == 4 and ds["test_dataset"] not in [lst.split("_")[0] for lst in ds["train_lists"]]:
            base_method = ds["method"].split("_no_es")[0] + "-mismatched-no_es"
            is_calibrated = False
        elif len(ds["train_lists"]) == 5:
            base_method = ds["method"].split("_no_es")[0] + "-all-no_es"
            is_calibrated = False
    elif not ds["method"].endswith("_plus_dp_cal"):
        if len(ds["train_lists"]) == 1 and ds["train_lists"][0].split("_")[0] == ds["test_dataset"]:
            base_method = ds["method"] + "-matched"
            is_calibrated = False
        elif len(ds["train_lists"]) == 4 and ds["test_dataset"] not in [lst.split("_")[0] for lst in ds["train_lists"]]:
            base_method = ds["method"] + "-mismatched"
            is_calibrated = False
        elif len(ds["train_lists"]) == 5:
            base_method = ds["method"] + "-all"
            is_calibrated = False
    return pd.Series({
        "base_method": base_method,
        "is_calibrated": is_calibrated,
        "seed": ds["seed"] if ds["seed"] != "all" else 0,
        "test_dataset": ds["test_dataset"],
        "result": ds["result"],
        "min_result": ds["min_result"],
    })

individuals = OrderedDict([
    ("zero_shot", {"label": "Zero-shot", "color": "black", "hatch": None, "alpha": 1.0,}),
    ("dp_calibration", {"label": "DP Calibration", "color": "black", "hatch": None, "alpha": 0.5,}),
    ("roberta-large-mnli", {"label": "RoBERTa-Large-MNLI", "color": "tab:red", "hatch": None, "alpha": 0.5,}),
    ("distilbert-base-uncased", {"label": "DistilBERT", "color": "tab:cyan", "hatch": None, "alpha": 0.5,}),
    # ("deberta-v2-xlarge", {"label": "DeBERTa-v2-XLarge", "color": "tab:purple", "hatch": None, "alpha": 0.5,}),
])

kwargs = OrderedDict([
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

def plot_bar(x, df, df_min, ax, bar_width, **kwargs):
    y = df["median"]
    y = 0 if len(y) == 0 else y.item()
    q1 = df["q1"]
    q1 = 0 if len(q1) == 0 else q1.item()
    q3 = df["q3"]
    q3 = 0 if len(q3) == 0 else q3.item()
    yerr = np.array([[y - q1], [q3 - y]])
    bar = ax.bar(x, y, bar_width, yerr=yerr, **kwargs)
    
    y = df_min["median"]
    y = 0 if len(y) == 0 else y.item()
    q1 = df_min["q1"]
    q1 = 0 if len(q1) == 0 else q1.item()
    q3 = df_min["q3"]
    q3 = 0 if len(q3) == 0 else q3.item()
    yerr = np.array([[y - q1], [q3 - y]])
    if kwargs["color"] in ["black", "k"]:
        color = "white"
    else:
        color = "k"
    ax.hlines(y, x - bar_width / 2, x + bar_width / 2, color=color, linestyle="-", linewidth=3)
    ax.errorbar(x, y, yerr=yerr, fmt="", color=color)



def plot_results(data, dataset_name, metric, filename, mode="mean"):
    agg = data.groupby(["test_dataset", "base_method", "is_calibrated"]).agg(
        median=("result", "median"),
        q1=("result", lambda x: np.quantile(x, 0.25)),
        q3=("result", lambda x: np.quantile(x, 0.75)),
    ).reset_index()
    
    agg_min = data.groupby(["test_dataset", "base_method", "is_calibrated"]).agg(
        median=("min_result", "median"),
        q1=("min_result", lambda x: np.quantile(x, 0.25)),
        q3=("min_result", lambda x: np.quantile(x, 0.75)),
    ).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bar_width = 0.05

    for i, method in enumerate(individuals.keys()):
        plot_bar(
            i * bar_width, 
            agg.loc[agg["base_method"] == method], 
            agg_min.loc[(agg_min["base_method"] == method)],
            ax, 
            bar_width,
            label=individuals[method]["label"], 
            color=individuals[method]["color"], 
            hatch=individuals[method]["hatch"], 
            alpha=individuals[method]["alpha"],
        )

    for i, method in enumerate(kwargs.keys()):
        plot_bar(
            i * bar_width * 2 + len(individuals) * bar_width, 
            agg.loc[(agg["base_method"] == method) & (~agg["is_calibrated"]),:],
            agg_min.loc[(agg_min["base_method"] == method) & (~agg_min["is_calibrated"]),:],
            ax, 
            bar_width,
            label=kwargs[method]["label"], 
            color=kwargs[method]["color"], 
            hatch=kwargs[method]["hatch"], 
            alpha=1.0,
        )

        plot_bar(
            i * bar_width * 2 + bar_width + len(individuals) * bar_width, 
            agg.loc[(agg["base_method"] == method) & (agg["is_calibrated"]),:],
            agg_min.loc[(agg_min["base_method"] == method) & (agg_min["is_calibrated"]),:],
            ax, 
            bar_width,
            color=kwargs[method]["color"], 
            hatch=kwargs[method]["hatch"], 
            alpha=0.6,
        )
        
    ax.set_title(f"{metric} on {dataset2name[dataset_name]['name']}")
    ax.set_ylabel(metric)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.grid(True)
    ylim = ax.get_ylim()
    ax.set_ylim(0, min(ylim[1], 1.2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def main(
    dataset: str,
    metric: str,
    results_path: str,
    output_dir: str,
    mode: str = "mean",
):
    # Read results
    results_path = Path(results_path)
    data = pd.read_json(results_path, lines=True)
    data = data[data["test_dataset"] == dataset]
    data = data[data["test_lst"].str.startswith("test_")]
    if len(data) == 0:
        return
    data = data.apply(parse_train_scenario, axis=1)

    # Plot results
    output_dir = Path(output_dir)
    plot_results(data, dataset, metric, output_dir / f"{dataset}_{metric}.png", mode = mode)

    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)