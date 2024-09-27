
import os
from datasets import load_from_disk
import pandas as pd
from pathlib import Path

def main(
    input_dir,
    output_dir,
    split,
):
    if split == "validation":
        split = "val"
    ds = load_from_disk(input_dir).with_format("numpy")
    index = ds["idx"].astype(int)
    for k in ds.features:
        if k == "idx":
            continue
        if k == "label":
            arr = ds[k].astype(int)
        else:
            arr = ds[k]
        d = pd.DataFrame(arr, index=index)
        if "embeddings" in k and (Path(output_dir) / f"{split}_{k}.csv").exists():
            os.remove(Path(output_dir) / f"{split}_{k}.csv")
        if "embeddings" in k:
            d.to_pickle(Path(output_dir) / f"{split}_{k}.pkl")
        else:
            d.to_csv(Path(output_dir) / f"{split}_{k}.csv", index=True, header=False)

if __name__ == "__main__":
    from fire import Fire
    Fire(main)