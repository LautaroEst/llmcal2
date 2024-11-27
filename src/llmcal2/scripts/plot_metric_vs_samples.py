
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .results_matched import dataset2name


dataset2sizes={
    "sst2": [16, 64, 1024],
    "agnews": [16, 64, 1024],
    "dbpedia": [28, 112, 1792],
    "20newsgroups": [40, 160, 2560],
    "banking77": [77, 308, 4928],
}

def plot(results, datasets, methods, metric, output_path):
    fig, ax = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4))
    for i, dataset in enumerate(datasets):
        for method in methods:
            df = results[(results["test_dataset"] == dataset) & (results["method"] == method)].copy()
            if len(df) == 0:
                continue
            mask = df["train_lists"].apply(lambda x: x == ["all"])
            if len(df[mask]) == len(df) == 1:
                df = pd.DataFrame({
                    "train_lists": [[f"{dataset}_{size}"] for size in dataset2sizes[dataset]],
                    "result": [df.loc[mask,"result"].item() for _ in range(len(dataset2sizes[dataset]))],
                })
            df.loc[:,"num_samples"] = df.apply(lambda x: int(x["train_lists"][0].split("_")[1]), axis=1)
            df = df.groupby("num_samples")["result"].agg(
                median=lambda x: x.median(),
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
            ).sort_index()
            ax[i].plot(df.index, df["median"], label=method, color=f"C{methods.index(method)}", marker="o", linestyle="-", linewidth=2, markersize=5)
            ax[i].fill_between(df.index, df["q1"], df["q3"], alpha=0.3, color=f"C{methods.index(method)}")
                
        ax[i].set_title(dataset)
        ax[i].set_ylim(0, 2)
        ax[i].grid(axis="y")
        

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(
    datasets,
    metric,
    methods,
    results_path,
    output_dir,
):
    datasets = list(map(str, datasets.split()))
    methods = list(map(str, methods.split()))
    output_dir = Path(output_dir)

    results = pd.read_json(results_path, orient="records", lines=True)
    mask = results.apply(
        lambda x: \
            (x["train_lists"][0].split("_")[0] == x["test_dataset"] or x["train_lists"][0] == "all") \
            and len(x["train_lists"]) == 1 \
            and x["test_lst"].startswith("test"), 
        axis=1
    )
    results = results.loc[mask,:]
    plot(results, datasets, methods, metric, output_dir / "samples.pdf")

if __name__ == "__main__":
    from fire import Fire
    Fire(main)