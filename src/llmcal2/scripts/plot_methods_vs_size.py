
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..evaluation.metrics import compute_metric

def load_data(results_dir, methods, datasets, model):

    data = []
    for model_dir in results_dir.iterdir():
        if model != model_dir.name:
            continue
        for method_dir in model_dir.iterdir():
            if method_dir.name not in methods:
                continue
            for dataset_dir in method_dir.iterdir():
                if dataset_dir.name not in datasets:
                    continue
                for size_dir in dataset_dir.iterdir():
                    size = size_dir.name.split("=")[1]
                    if size != "all":
                        size = int(size)
                    for seed_dir in size_dir.iterdir():
                        seed = seed_dir.name.split("=")[1]
                        if seed != "all":
                            seed = int(seed)
                        if (test_logits := seed_dir / "test_logits.csv").exists():
                            logits_path = str(test_logits)
                        else:
                            continue
                        if (test_label := seed_dir / "test_label.csv").exists():
                            labels_path = str(test_label)
                        else:
                            continue
                        data.append({
                            "method": method_dir.name,
                            "dataset": dataset_dir.name,
                            "size": size,
                            "seed": seed,
                            "logits": logits_path,
                            "labels": labels_path,
                        })
    return pd.DataFrame(data)


def compute_metrics(data, metrics):
    data_with_metrics = data.copy()
    for metric in metrics:
        data_with_metrics[f"metric:{metric}"] = None
    for i, row in tqdm(data.iterrows(), total=len(data)):
        logits = pd.read_csv(row["logits"], index_col=0, header=None).values.astype(float)
        labels = pd.read_csv(row["labels"], index_col=0, header=None).values.flatten().astype(int)
        for metric in metrics:
            data_with_metrics.loc[i, f"metric:{metric}"] = compute_metric(logits, labels, metric)
    data_with_metrics = data_with_metrics.drop(columns=["logits", "labels"])
    return data_with_metrics
            

kwargs = {
    "no_adaptation": {"label": "No adaptation", "color": "black", "ls": "--"},
    "dp_calibration": {"label": "DP calibration", "color": "tab:blue", "ls": "--", "marker": "*"},
    "temp_scaling": {"label": "Temperature scaling", "color": "tab:orange", "ls": "--", "marker": "*"},
    "bias_only": {"label": "Bias only", "color": "tab:green", "ls": "--", "marker": "*"},
    "lora": {"label": "LoRA", "color": "tab:red", "ls": "--", "marker": "*"},
    "lora_norm": {"label": "LoRA CE", "color": "tab:purple", "ls": "--", "marker": "*"},
    "lora_no_es": {"label": "LoRA (no ES)", "color": "tab:red", "ls": "--", "marker": "*"},
    "lora_plus_dp_calibration_no_es": {"label": "LoRA + DP calibration (no ES)", "color": "tab:blue", "ls": "-", "marker": "*"},
    "lora_plus_temp_scaling_no_es": {"label": "LoRA + Temp Scaling (no ES)", "color": "tab:orange", "ls": "-", "marker": "*"},
    "lora_norm_plus_dp_calibration_no_es": {"label": "LoRA CE + DP calibration (no ES)", "color": "tab:blue", "ls": ":", "marker": "*"},
    "lora_norm_plus_temp_scaling_no_es": {"label": "LoRA CE + Temp Scaling (no ES)", "color": "tab:orange", "ls": ":", "marker": "*"},
}

metric2name = {
    "ner": "NER",
    "nce": "NCE",
    "nbrier": "NBrier",
    "cal_loss_nce": "Calibration loss (NCE)",
    "ece": "ECE",
}

dataset2name = {
    "sst2": {"name": "SST-2", "num_classes": 2},
    "agnews": {"name": "AG News", "num_classes": 4},
    "dbpedia": {"name": "DBpedia", "num_classes": 14},
    "20newsgroups": {"name": "20 Newsgroups", "num_classes": 20},
    "banking77": {"name": "Banking77", "num_classes": 77},
}

def plot_and_save(data, metrics, datasets, methods, filename):

    # Create axes
    fig, ax = plt.subplots(len(metrics),len(datasets), figsize=(len(datasets)*2, len(metrics)*2), sharex="col")
    if len(metrics) == 1 and len(datasets) == 1:
        ax = np.array([[ax]])
    elif len(metrics) == 1:
        ax = ax[np.newaxis, :]
    elif len(datasets) == 1:
        ax = ax[:, np.newaxis]

    # Compute statistics
    data = data.copy()
    stats = [("median", np.median), ("q1", lambda x: np.quantile(x, 0.25)), ("q3", lambda x: np.quantile(x, 0.75))]
    data = data.drop(columns=["seed"]).groupby(["method", "dataset", "size"]).agg(**{
        f"{metric}-{stat}": (f"metric:{metric}", fn) for metric in metrics for stat, fn in stats
    })

    for i, dataset in enumerate(datasets):
        dataset_sizes = data.loc[(slice(None), dataset, slice(None)), :].index.get_level_values("size").unique().values
        dataset_sizes = np.sort(dataset_sizes[dataset_sizes != "all"]).astype(int)
        min_size = np.min(dataset_sizes)
        max_size = np.max(dataset_sizes)

        for j, metric in enumerate(metrics):
            if i == 0:
                ax[j,0].set_ylabel(metric2name[metric],fontsize=16)
            for m, method in enumerate(methods):
                try:
                    portion = data.loc[(method, dataset, slice(None)), [f"{metric}-{stat[0]}" for stat in stats]].copy().reset_index()
                except KeyError:
                    print(f"Skipping {method} for {dataset} and {metric}")
                    continue
                
                portion = portion.set_index("size").drop(columns=["method", "dataset"])
                if method == "no_adaptation":
                    x = [min_size, max_size]
                    y = [portion.loc["all",f"{metric}-median"], portion.loc["all",f"{metric}-median"]]
                    ax[j,i].plot(x, y, **kwargs[method])
                else:
                    portion.index = portion.index.astype(int)
                    portion = portion.sort_index()
                    ax[j,i].plot(portion.index, portion[f"{metric}-median"], **kwargs[method])
                
            ax[j,i].yaxis.grid(True)
            ax[j,i].set_xscale("log")
            ax[j,i].set_xticks([])
            ax[j,i].minorticks_off()
            ylim = ax[j,i].get_ylim()
            # sup = 1.2 * data.loc[(slice(None), dataset, slice(None)), f"{metric}-median"].max()
            sup = 1.5
            ax[j,i].set_ylim(max(0,ylim[0]), min(sup, ylim[1]))

        ax[0,i].set_title(f"{dataset2name[dataset]['name']}",fontsize=14)
        # ax[-1,i].set_xlim(0.8 * min(dataset_sizes), 1.2 * max(dataset_sizes))
        ax[-1,i].set_xticks(dataset_sizes)
        ax[-1,i].set_xticklabels(dataset_sizes, fontsize=10)
        ax[-1,i].set_xlabel(f"(x{dataset2name[dataset]['num_classes']})", fontsize=12)
    
    hand, lab = ax[0,0].get_legend_handles_labels()
    fig.legend(hand, lab, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=max(1,len(methods)//2), fancybox=True, shadow=True, fontsize=8)
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def main(
    model,
    metrics,
    datasets,
    methods,
    results_dir,
    output_dir,
    overwrite = False,
):
    if isinstance(metrics, str) and "," in metrics:
        metrics = [m for m in metrics.split(",") if m != ""]
    if isinstance(datasets, str) and "," in datasets:
        datasets = [d for d in datasets.split(",") if d != ""]
    if isinstance(methods, str) and "," in methods:
        methods = [m for m in methods.split(",") if m != ""]
        
    output_dir = Path(output_dir)
    results_dir = Path(results_dir)


    if overwrite:
        # Load data
        data = load_data(results_dir, methods, datasets, model)
        
        # Compute metrics
        data_with_metrics = compute_metrics(data, metrics)
        data_with_metrics.to_csv(output_dir / "results.csv", index=False)
    
    elif not (output_dir / "results.csv").exists():
        raise ValueError("Output file does not exist. Set overwrite to True to create it.")
    
    else:
        data_with_metrics = pd.read_csv(output_dir / "results.csv", header=0, index_col=None)
    
    # Plot
    plot_and_save(data_with_metrics, metrics, datasets, methods, output_dir / f"{model}.pdf")
    



if __name__ == '__main__':
    from fire import Fire
    Fire(main)