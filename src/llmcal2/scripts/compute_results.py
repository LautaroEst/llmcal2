
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..evaluation.metrics import compute_metric


def read_results(root_results_dir: Path):
    data = []
    for method_dir in root_results_dir.iterdir():
        method = method_dir.name
        for train_list_dir in method_dir.iterdir():
            train_list_id = train_list_dir.name
            train_lists = train_list_id.split("__")
            if len(set([lst.split("_")[-1] for lst in train_lists])) != 1:
                raise ValueError(f"seed is not unique across training lists")
            seed = int(train_lists[0].split("_")[-1]) if not train_lists[0] == "all" else "all"
            for test_dir in train_list_dir.iterdir():
                test_dataset = test_dir.name
                if not test_dataset.startswith("test="):
                    continue
                test_dataset = test_dataset.split("=")[1]
                for test_list_dir in test_dir.iterdir():
                    test_list = test_list_dir.name.split("=")[1]
                    if not (labels_path := test_list_dir / "labels.csv").exists():
                        continue
                    if not (logits_path := test_list_dir / "logits.csv").exists():
                        continue
                    data.append({
                        "method": method,
                        "train_lists": train_lists,
                        "seed": seed,
                        "test_dataset": test_dataset,
                        "test_lst": test_list,
                        "logits": logits_path,
                        "labels": labels_path,
                    })
    return pd.DataFrame(data)
                        

def compute_metrics(data, metric):
    data_with_metrics = data.copy()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        logits = pd.read_csv(row["logits"], index_col=0, header=None).values.astype(float)
        labels = pd.read_csv(row["labels"], index_col=0, header=None).values.flatten().astype(int)
        if row["method"] == "lora_fs" and (row["train_lists"] == ["agnews_1024_0.3_0"] or row["train_lists"] == ["agnews_1024_0.3_1"]):
            from scipy.special import log_softmax
            import numpy as np
            scores = -log_softmax(logits, axis=1)[np.arange(len(labels)), labels]
            print(scores)
            
        value = compute_metric(logits, labels, metric)
        data_with_metrics.loc[i, "result"] = value
    data_with_metrics = data_with_metrics.drop(columns=["logits", "labels"])
    return data_with_metrics


def main(
    metric: str,
    root_results_dir: str,
    output_path: str,
):
    # Read results
    root_results_dir = Path(root_results_dir)
    data = read_results(root_results_dir)
    
    # Compute metrics
    data_with_metrics = compute_metrics(data, metric)
    data_with_metrics.to_json(output_path, orient="records", lines=True)
    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)