
from pathlib import Path
import pandas as pd

def main():
    all_datasets = ["sst2", "agnews", "dbpedia", "20newsgroups", "banking77"]
    data = []
    for method in Path("outputs/adaptation/llama3.2-1b").iterdir():
        if method.name not in ["lora_ans", "lora_fs"]:
            continue
        for train_list_dir in method.iterdir():
            train_list = train_list_dir.name.split("__")
            if len(train_list) == 1:
                dataset = train_list[0].split("_")[0]
                version = "matched"
            elif len(train_list) == 4:
                dataset = (set(all_datasets) - set([l.split("_")[0] for l in train_list])).pop()
                version = "mismatched"
            elif len(train_list) == 5:
                version = "all"
                dataset = "all"
            seed = train_list[0].split("_")[-1]
            df = pd.read_csv(train_list_dir / "logs/metrics.csv", header=0, index_col=None)
            data.append({
                "dataset": dataset,
                "method": method.name,
                "version": version,
                "seed": seed,
                "max_step": df["step"].max(),
            })
    df = pd.DataFrame(data).sort_values(["dataset","method", "version", "seed", "max_step"])
    df.to_csv("outputs/adaptation/llama3.2-1b/echo_early_stopped.csv", index=False)

    


        


if __name__ == '__main__':
    main()