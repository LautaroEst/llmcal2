
from pathlib import Path
from typing import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kwargs = OrderedDict([
    ("no_adaptation", {"label": "No Adaptation", "color": "black", "hbar_color": "gray", "linewidth": 4}),
    ("lora_fs", {"label": "LORA-FS", "color": "tab:blue", "hbar_color": "lightblue", "linewidth": 4}),
    ("lora_ans", {"label": "LORA-ANS", "color": "tab:orange", "hbar_color": "moccasin", "linewidth": 4}),
    ("lora_norm-5", {"label": "LORA-Norm (K=5)", "color": "tab:green", "hbar_color": "lightgreen", "linewidth": 4}),
])

dataset2name = {
    "sst2": {"name": "SST-2", "num_classes": 2},
    "agnews": {"name": "AG News", "num_classes": 4},
    "dbpedia": {"name": "DBpedia", "num_classes": 14},
    "20newsgroups": {"name": "20 Newsgroups", "num_classes": 20},
    "banking77": {"name": "Banking77", "num_classes": 77},
}


def plot_train_test_on_same_dataset(data: pd.DataFrame, psr, datasets, methods, num_train_samples, test_lst, filename):
    data = data.copy()
    data = data[
        data["num_train_samples"].isin([num_train_samples, "all"]) & (data["test_lst"] == test_lst)
    ]
    data = data[data["train_dataset"] == data["test_dataset"]]
    data["dataset"] = data["train_dataset"]
    data = data[data["dataset"].isin(datasets)]
    data = data.drop(columns=["train_dataset", "test_dataset", "num_train_samples", "test_lst"])
    
    grouped = data.groupby(['dataset', 'method'])['loss'].agg(
        mean='mean',
        std='std',
    ).reset_index()
    loss = grouped.pivot(index='dataset', columns='method', values='mean')
    error_bars = grouped.pivot(index='dataset', columns='method', values='std')

    grouped = data.groupby(['dataset', 'method'])['cal_loss'].agg(
        mean='mean',
        std='std',
    ).reset_index()
    cal_loss = grouped.pivot(index='dataset', columns='method', values='mean')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(loss))
    for i, method in enumerate(kwargs.keys()):
        if method not in methods:
            continue
        ax.bar(index + i * bar_width, loss[method], bar_width, label=kwargs[method]["label"], yerr=error_bars[method], color=kwargs[method]["color"])
        ax.hlines(cal_loss[method], index + i * bar_width - bar_width / 2, index + i * bar_width + bar_width / 2, color=kwargs[method]["hbar_color"], linewidth=kwargs[method]["linewidth"])
        
    ax.set_xticks(index + bar_width * (len(kwargs) - 1) / 2)
    ax.set_xticklabels([dataset2name[dataset]["name"] for dataset in loss.index], fontsize=14)
    ax.set_title(f"PSR: {psr}")
    ax.grid(axis="y")
    ax.legend()
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def main(
    psr: str,
    datasets: str,
    methods: str,
    results_path: str,
    output_dir: str,
):
    output_dir = Path(output_dir)
    results_path = Path(results_path)
    data_with_metrics = pd.read_csv(results_path)

    for size in set(data_with_metrics["num_train_samples"].unique()) - {"all"}:
        valid_test_lsts = data_with_metrics.loc[data_with_metrics["num_train_samples"] == size, "test_lst"].unique()
        for test_lst in valid_test_lsts:
            plot_train_test_on_same_dataset(data_with_metrics, psr, datasets, methods, size, test_lst, output_dir / f"samedataset--{size}--{test_lst}.pdf")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)