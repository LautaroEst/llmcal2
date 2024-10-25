
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..evaluation.metrics import compute_psr_with_mincal


def read_results(root_results_dir: Path, datasets: list, methods: list):
    data = []
    for method_dir in root_results_dir.iterdir():
        method = method_dir.name
        if method not in methods:
            continue
        if method == "no_adaptation":
            for train_dataset_dir in method_dir.iterdir():
                train_dataset = train_dataset_dir.name
                test_dataset = train_dataset_dir.name
                for test_lst_dir in train_dataset_dir.iterdir():
                    test_lst = test_lst_dir.name
                    if not (labels_path := test_lst_dir / "labels.csv").exists():
                        continue
                    if not (logits_path := test_lst_dir / "logits.csv").exists():
                        continue
                    data.append({
                        "method": method,
                        "train_dataset": train_dataset,
                        "num_train_samples": "all",
                        "test_dataset": test_dataset,
                        "test_lst": test_lst,
                        "seed": "all",
                        "logits": logits_path,
                        "labels": labels_path,
                    })
        else:
            for train_dataset_dir in method_dir.iterdir():
                train_dataset = train_dataset_dir.name.split("=")[1]
                if train_dataset not in datasets:
                    continue
                for train_lst_dir in train_dataset_dir.iterdir():
                    train_lst = train_lst_dir.name.split("=")[1]
                    _, size, num_seed = train_lst.split("_")
                    for test_dataset_dir in train_lst_dir.iterdir():
                        if not test_dataset_dir.name.startswith("test="):
                            continue
                        test_dataset = test_dataset_dir.name.split("=")[1]
                        if test_dataset not in datasets:
                            continue
                        for test_lst_dir in test_dataset_dir.iterdir():
                            test_lst = test_lst_dir.name.split("=")[1]

                            if not (labels_path := test_lst_dir / "labels.csv").exists():
                                continue
                            if not (logits_path := test_lst_dir / "logits.csv").exists():
                                continue
                            
                            data.append({
                                "method": method,
                                "train_dataset": train_dataset,
                                "num_train_samples": size,
                                "test_dataset": test_dataset,
                                "test_lst": test_lst,
                                "seed": num_seed,
                                "logits": logits_path,
                                "labels": labels_path,
                            })
    return pd.DataFrame(data)
                        

def compute_metrics(data, psr):
    data_with_metrics = data.copy()
    psr, mode = psr.split("_")
    for i, row in tqdm(data.iterrows(), total=len(data)):
        logits = pd.read_csv(row["logits"], index_col=0, header=None).values.astype(float)
        labels = pd.read_csv(row["labels"], index_col=0, header=None).values.flatten().astype(int)
        loss, cal_loss = compute_psr_with_mincal(logits, labels, psr, mode)
        data_with_metrics.loc[i, "loss"] = loss
        data_with_metrics.loc[i, "cal_loss"] = cal_loss
    data_with_metrics = data_with_metrics.drop(columns=["logits", "labels"])
    return data_with_metrics


def main(
    psr: str,
    datasets: str,
    methods: str,
    root_results_dir: str,
    output_path: str,
):
    root_results_dir = Path(root_results_dir)
    output_dir = Path(output_path)

    if isinstance(datasets, str) and "," in datasets:
        datasets = [d for d in datasets.split(",") if d != ""]
    if isinstance(methods, str) and "," in methods:
        methods = [m for m in methods.split(",") if m != ""]
    
    # Load data
    data = read_results(root_results_dir, datasets, methods)
    
    # Compute metrics
    data_with_metrics = compute_metrics(data, psr)
    data_with_metrics.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)