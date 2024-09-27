
import numpy as np
import datasets
datasets.disable_caching()

def load_sst2():
    data = datasets.load_dataset("nyu-mll/glue", "sst2")
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").rename(columns={"sentence": "text"}),
        "test": data["validation"].to_pandas().set_index("idx").rename(columns={"sentence": "text"}),
    }
    return datadict


def load_agnews():
    data = datasets.load_dataset("ag_news")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"]))).to_pandas().set_index("idx")
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"]))).to_pandas().set_index("idx")
    datadict = {
        "train": data["train"],
        "test": data["test"]
    }
    return datadict


def load_dbpedia():
    data = datasets.load_dataset("fancyzhx/dbpedia_14")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    rs = np.random.RandomState(27834)
    test_idx = rs.choice(len(data["test"]), 7000, replace=False)
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"}),
        "test": data["test"].select(test_idx).to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"}),
    }
    return datadict


def load_newsgroups():
    data = datasets.load_dataset("SetFit/20_newsgroups")
    classes_names = data["train"].to_pandas().loc[:,["label","label_text"]].drop_duplicates().set_index("label").squeeze().sort_index().tolist()
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))
    for split in data:
        features = data[split].features
        features["label"] = datasets.ClassLabel(
            num_classes=20, 
            names=classes_names)
        data[split] = data[split].cast(features)
    
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").drop(columns=["label_text"]),
        "test": data["test"].to_pandas().set_index("idx").drop(columns=["label_text"]),
    }

    return datadict


def load_banking():
    data = datasets.load_dataset("PolyAI/banking77")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    all_data = datasets.concatenate_datasets([data["train"], data["test"]])

    rs = np.random.RandomState(7348)
    idx = rs.permutation(len(all_data))
    train_idx = idx[:len(data["train"])]
    test_idx = idx[len(data["train"]):]
    
    datadict = {
        "train": all_data.select(train_idx).to_pandas().set_index("idx"),
        "test": all_data.select(test_idx).to_pandas().set_index("idx"),
    }
    
    return datadict


def main(dataset, output_dir):
    if dataset == "sst2":
        datadict = load_sst2()
    elif dataset == "agnews":
        datadict = load_agnews()
    elif dataset == "dbpedia":
        datadict = load_dbpedia()
    elif dataset == "20newsgroups":
        datadict = load_newsgroups()
    elif dataset == "banking77":
        datadict = load_banking()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    for split, df in datadict.items():
        df.to_csv(f"{output_dir}/{split}.csv", index=True, header=True)
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)