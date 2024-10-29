
from pathlib import Path
from typing import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

kwargs = OrderedDict([
    ("no_adaptation", {"label": "Zero-shot", "color": "black", "hbar_color": "gray", "linewidth": 4}),
    ("sst2", {"label": "Trained on SST-2", "color": "tab:blue", "hbar_color": "lightblue", "linewidth": 4}),
    ("agnews", {"label": "Trained on AG News", "color": "tab:orange", "hbar_color": "moccasin", "linewidth": 4}),
    ("dbpedia", {"label": "Trained on DBpedia", "color": "tab:green", "hbar_color": "lightgreen", "linewidth": 4}),
    ("20newsgroups", {"label": "Trained on 20 Newsgroups", "color": "tab:red", "hbar_color": "lightcoral", "linewidth": 4}),
    ("banking77", {"label": "Trained on Banking77", "color": "tab:purple", "hbar_color": "thistle", "linewidth": 4}),
])

dataset2name = {
    "sst2": {"name": "SST-2", "num_classes": 2},
    "agnews": {"name": "AG News", "num_classes": 4},
    "dbpedia": {"name": "DBpedia", "num_classes": 14},
    "20newsgroups": {"name": "20 Newsgroups", "num_classes": 20},
    "banking77": {"name": "Banking77", "num_classes": 77},
}


def plot_train_crosstalk_dataset(data: pd.DataFrame, psr, datasets, method, filename):
    data = data.copy()
    data = data[data["method"].isin([method, "no_adaptation"])]
    data = data[data["train_dataset"].isin(datasets)]
    data = data[data["test_dataset"].isin(datasets)]
    data.loc[data["method"] == "no_adaptation", "train_dataset"] = "no_adaptation"
    data = data.drop(columns=["method", "num_train_samples", "test_lst"])
    
    grouped = data.groupby(['train_dataset', 'test_dataset'])['loss'].agg(
        mean='mean',
        std='std',
    ).reset_index()
    loss = grouped.pivot(index='test_dataset', columns='train_dataset', values='mean')
    error_bars = grouped.pivot(index='test_dataset', columns='train_dataset', values='std')

    grouped = data.groupby(['train_dataset', 'test_dataset'])['cal_loss'].agg(
        mean='mean',
        std='std',
    ).reset_index()
    cal_loss = grouped.pivot(index='test_dataset', columns='train_dataset', values='mean')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bar_width = 0.2

    # Plot no adaptation on first column
    index = np.arange(len(loss))
    ax.bar(index, loss["no_adaptation"], bar_width, label=kwargs["no_adaptation"]["label"], yerr=error_bars["no_adaptation"], color=kwargs["no_adaptation"]["color"])
    ax.hlines(cal_loss["no_adaptation"], index - bar_width / 2, index + bar_width / 2, color=kwargs["no_adaptation"]["hbar_color"], linewidth=kwargs["no_adaptation"]["linewidth"])
    
    # Plot the rest
    for i, train_dataset in enumerate(datasets,1):
        ax.bar(index + i * bar_width, loss[train_dataset], bar_width, label=kwargs[train_dataset]["label"], yerr=error_bars[train_dataset], color=kwargs[train_dataset]["color"])
        ax.hlines(cal_loss[train_dataset], index + i * bar_width - bar_width / 2, index + i * bar_width + bar_width / 2, color=kwargs[train_dataset]["hbar_color"], linewidth=kwargs[train_dataset]["linewidth"])
        
    ax.set_ylim(0, 2)
    ax.set_xticks(index + bar_width * (len(datasets) - 1) / 2)
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

    for method in set(methods.split(",")) - {"no_adaptation"}:
        plot_train_crosstalk_dataset(data_with_metrics, psr, datasets, method, output_dir / f"crosstalk--{method}.pdf")


if __name__ == '__main__':
    from fire import Fire
    Fire(main)