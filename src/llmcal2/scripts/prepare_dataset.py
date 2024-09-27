
import numpy as np
import datasets
datasets.disable_caching()

def load_sst2(train_list_path, test_list_path):
    data = datasets.load_dataset("nyu-mll/glue", "sst2")
    train_list = np.loadtxt(train_list_path, dtype=int)
    test_list = np.loadtxt(test_list_path, dtype=int)
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").rename(columns={"sentence": "text"}).loc[train_list],
        "test": data["validation"].to_pandas().set_index("idx").rename(columns={"sentence": "text"}).loc[test_list]
    }
    return datadict


def load_agnews(train_list_path, test_list_path):
    data = datasets.load_dataset("ag_news")
    train_list = np.loadtxt(train_list_path, dtype=int)
    test_list = np.loadtxt(test_list_path, dtype=int)
    datadict = {
        "train": data["train"].add_column("idx", np.arange(len(data["train"]))).to_pandas().set_index("idx").loc[train_list],
        "test": data["test"].add_column("idx", np.arange(len(data["test"]))).to_pandas().set_index("idx").loc[test_list]
    }
    return datadict


def load_dbpedia(train_list_path, test_list_path):
    data = datasets.load_dataset("fancyzhx/dbpedia_14")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    train_list = np.loadtxt(train_list_path, dtype=int)
    test_list = np.loadtxt(test_list_path, dtype=int)
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").loc[train_list,["content","label"]].rename(columns={"content": "text"}),
        "test": data["test"].to_pandas().set_index("idx").loc[test_list,["content","label"]].rename(columns={"content": "text"}),
    }
    return datadict


def load_newsgroups(train_list_path, test_list_path):
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
    
    train_list = np.loadtxt(train_list_path, dtype=int)
    test_list = np.loadtxt(test_list_path, dtype=int)
    datadict = {
        "train": data["train"].to_pandas().set_index("idx").drop(columns=["label_text"]).loc[train_list],
        "test": data["test"].to_pandas().set_index("idx").drop(columns=["label_text"]).loc[test_list],
    }

    return datadict


def load_banking(train_list_path, test_list_path):
    data = datasets.load_dataset("PolyAI/banking77",trust_remote_code=True)
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    all_data = datasets.concatenate_datasets([data["train"], data["test"]])

    train_list = np.loadtxt(train_list_path, dtype=int)
    test_list = np.loadtxt(test_list_path, dtype=int)
    datadict = {
        "train": all_data.to_pandas().set_index("idx").loc[train_list],
        "test": all_data.to_pandas().set_index("idx").loc[test_list],
    }
    
    return datadict


def main(dataset, lists_dir, output_dir):
    train_list_path = f"{lists_dir}/train.txt"
    test_list_path = f"{lists_dir}/test.txt"

    if dataset == "sst2":
        datadict = load_sst2(train_list_path, test_list_path)
    elif dataset == "agnews":
        datadict = load_agnews(train_list_path, test_list_path)
    elif dataset == "dbpedia":
        datadict = load_dbpedia(train_list_path, test_list_path)
    elif dataset == "20newsgroups":
        datadict = load_newsgroups(train_list_path, test_list_path)
    elif dataset == "banking77":
        datadict = load_banking(train_list_path, test_list_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    for split, df in datadict.items():
        df.to_csv(f"{output_dir}/{split}.csv", index=True, header=True)
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)