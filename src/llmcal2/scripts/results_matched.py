# Plot results matched data


from itertools import product
from pathlib import Path
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
    methods,
    cal_methods,
    datasets,
    metric,
    results_path,
    outputs_dir,
):
    methods = list(map(str, methods.split()))
    cal_methods = [""] + list(map(str, cal_methods.split()))
    fullname_methods = {m + cm: (m, cm) for m, cm in product(methods, cal_methods)}
    datasets = list(map(str, datasets.split()))
    outputs_dir = Path(outputs_dir)

    results = pd.read_json(results_path, orient="records", lines=True)
    results = results[results["method"].isin(fullname_methods.keys()) & results["test_dataset"].isin(datasets)]
    mask = results.apply(
        lambda x: \
            (x["train_lists"][0].split("_")[0] == x["test_dataset"] or x["train_lists"][0] == "all") \
            and len(x["train_lists"]) == 1 \
            and x["test_lst"].startswith("test") \
            and (x["train_lists"][0] == "all" or x["train_lists"][0].split("_")[1] == str(num_samples[x["test_dataset"]])),
        axis=1
    )
    results = results.loc[mask,:]
    create_table(results.copy(), fullname_methods, methods, cal_methods, datasets, outputs_dir / "results.tex")
    create_plot(results.copy(), fullname_methods, methods, cal_methods, datasets, metric, outputs_dir / "results.pdf")


def create_table(results, fullname_methods, methods, cal_methods, datasets, output_path):
    results = results.loc[:,["method", "test_dataset", "seed", "result"]]
    results.loc[results["seed"] == "all","seed"] = 0
    results["seed"] = results["seed"].astype(int)
    results = results.groupby(["method", "test_dataset"]).agg({
        "result": ["mean", "std"]
    }).reset_index()
    results.columns = ["method", "test_dataset", "mean", "std"]
    results = results.apply(lambda x: format_results(x,fullname_methods), axis=1)
    results = results.loc[:,["base_method", "cal_method", "test_dataset", "formatted_result"]]
    results = results.set_index(["base_method", "cal_method"]).sort_index()
    results = results.pivot(columns="test_dataset", values="formatted_result")
    results = results.loc[[(m, cm) for m in methods for cm in cal_methods], datasets]
    results = results.rename(index=method2name)
    results = results.rename(columns=dataset2name)
    results.index.name = "Method"
    table = results.to_latex(escape=True)
    with open(output_path, "w") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\begin{tabular}{l" + "c" * len(datasets) + "}\n")
        f.write(table)
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{document}" + "\n")

def create_plot(results, fullname_methods, methods, cal_methods, datasets, metric, output_path):
    results = results.loc[:,["method", "test_dataset", "seed", "result"]]
    results.loc[results["seed"] == "all","seed"] = 0
    results["seed"] = results["seed"].astype(int)
    results = results.groupby(["method", "test_dataset"]).agg({
        "result": ["mean", "std"]
    }).reset_index()
    results.columns = ["method", "test_dataset", "mean", "std"]
    results = results.apply(lambda x: format_results(x,fullname_methods), axis=1)
    results = results.loc[:,["base_method", "cal_method", "test_dataset", "mean", "std"]]
    results = results.set_index("test_dataset").loc[datasets,:]
    results = results.rename(index=dataset2name)
    fig, ax = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 6))
    bar_width = 1
    for i, dataset in enumerate(datasets):
        data = results.loc[dataset2name[dataset],:]
        data = data.reset_index(drop=True).set_index(["base_method", "cal_method"])
        for j, method in enumerate(methods):
            for k, cal_method in enumerate(cal_methods):
                if cal_method == "":
                    label = method2name[method]
                else:
                    label = f"{method2name[method]} {method2name[cal_method]}"
                mean = data.loc[(method, cal_method), "mean"]
                std = data.loc[(method, cal_method), "std"]
                x = j * len(cal_methods) + k
                ax[i].bar(x, mean, yerr=std, color=f"C{j}", label=label, width=bar_width, alpha=(len(cal_methods)-k)/len(cal_methods)*0.8+0.2)
        ax[i].set_title(dataset2name[dataset])
        ax[i].grid(axis="y")
        ax[i].set_xticks([])
        ax[i].set_xticklabels([])
        ax[i].set_ylim(0, 1)
    
    ax[0].set_ylabel(metric2name[metric])
    ax[-1].legend(loc="upper left", bbox_to_anchor=(0.2, 1), ncol=1, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
                           
    



if __name__ == "__main__":
    from fire import Fire
    Fire(main)