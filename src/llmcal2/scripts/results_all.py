# Plot results matched data


from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def format_results(row, fullname_methods):
    row["base_method"] = fullname_methods[row["method"]][0]
    row["cal_method"] = fullname_methods[row["method"]][1]
    row["formatted_result"] = f"{row['mean']:.2f} ± {row['std']:.2f}"
    row = row[["base_method", "cal_method", "test_dataset", "mean", "std", "formatted_result"]]
    return row
    
method2name = {
    "no_adaptation": "No Adaptation",
    "lora_fs": "LoRA-FS",
    "lora_ans": "LoRA-ANS",
    "_plus_dp_cal": "+ DP Calibration",
}

dataset2name = {
    "sst2": "SST-2",
    "agnews": "AGNews",
    "dbpedia": "DBpedia",
    "20newsgroups": "20Newsgroups",
    "banking77": "Banking77",
}

metric2name = {
    "nce": "NCE",
}

num_samples = {
    "sst2": 1024,
    "agnews": 1024,
    "dbpedia": 1792,
    "20newsgroups": 2560,
    "banking77": 4928,
}

def main(
    baseline,
    models,
    cal_method,
    datasets,
    metric,
    results_path,
    outputs_dir,
):
    models = list(map(str, models.split()))
    datasets = list(map(str, datasets.split()))
    outputs_dir = Path(outputs_dir)

    results = pd.read_json(results_path, orient="records", lines=True)
    results = results.loc[results["test_lst"].str.startswith("test_")]
    create_plot(results.copy(), baseline, models, cal_method, datasets, outputs_dir / "results.pdf")

def create_plot(results, baseline, models, cal_method, datasets, output_path):
    all_models = [baseline] + models
    
    all_results = {"no_cal": [], "cal": [], "min_cal": []}
    for model in all_models:
        data = results.copy()
        if "_few_shot" not in model and "lora" in model:
            data = data.loc[data.apply(lambda x: all([int(s.split("_")[1]) == num_samples[x["test_dataset"]] for s in x["train_lists"]]))]
        elif "_few_shot" in model and "lora" in model:
            data = data.loc[data["train_lists"].str.len() == 5]
            model, n_shots = model.split("_few_shot")
            model = model + "_few_shot"
            n_shots = int(n_shots)
            data = data.loc[data.apply(lambda x: x["train_lists"][-1].startswith(f"{n_shots}shots"), axis=1)]
        elif "_few_shot" in model:
            model, n_shots = model.split("_few_shot")
            model = model + "_few_shot"
            n_shots = int(n_shots)
            data = data.loc[data.apply(lambda x: x["train_lists"][-1].startswith(f"{n_shots}shots"), axis=1)]

        import pdb; pdb.set_trace()
        no_cal = data.loc[data["method"] == model,["test_dataset", "seed", "result"]].groupby("test_dataset")["result"].agg(median=lambda x: x.median(), q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).loc[datasets]
        cal = data.loc[data["method"] == model + cal_method,["test_dataset", "seed", "result"]].groupby("test_dataset")["result"].agg(median=lambda x: x.median(), q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).loc[datasets]
        min_cal = data.loc[data["method"] == model,["test_dataset", "seed", "min_result"]].groupby("test_dataset")["min_result"].agg(median=lambda x: x.median(), q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).loc[datasets]
        all_results["no_cal"].append(no_cal)
        all_results["cal"].append(cal)
        all_results["min_cal"].append(min_cal)
       
    
    fig, ax = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4))
    bar_width = 1.
    bar_sep = 0.5
    for i, dataset in enumerate(datasets):

        for j, model in enumerate(all_models):
            x = j * (len(all_models) + bar_sep)
            y = all_results["no_cal"][j].loc[dataset, "median"]
            yerr = np.array([y - all_results["no_cal"][j].loc[dataset, "q1"], all_results["no_cal"][j].loc[dataset, "q3"] - y]).reshape(2, 1)
            ax[i].bar(x, y,yerr=yerr, label=model, color=f"C{j}", alpha=1, width=bar_width)
            
            x = j * (len(all_models) + bar_sep) + 1 * bar_width
            y = all_results["cal"][j].loc[dataset, "median"]
            yerr = np.array([y - all_results["cal"][j].loc[dataset, "q1"], all_results["cal"][j].loc[dataset, "q3"] - y]).reshape(2, 1)
            ax[i].bar(x, y, yerr=yerr, color=f"C{j}", alpha=0.7, width=bar_width)

            x = j * (len(all_models) + bar_sep) + 2 * bar_width
            y = all_results["min_cal"][j].loc[dataset, "median"]
            yerr = np.array([y - all_results["min_cal"][j].loc[dataset, "q1"], all_results["min_cal"][j].loc[dataset, "q3"] - y]).reshape(2, 1)
            ax[i].bar(x, y, yerr=yerr, color=f"C{j}", alpha=0.4, width=bar_width)

        ax[i].set_title(dataset2name[dataset])
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])
        ax[i].set_ylim(0, 1)
        ax[i].grid(axis="y")

    ax[-1].legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

                           
    



if __name__ == "__main__":
    from fire import Fire
    Fire(main)