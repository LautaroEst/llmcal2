
from pathlib import Path
import pandas as pd
import numpy as np
import datasets
datasets.disable_caching()

def create_lists_sst2(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("nyu-mll/glue", "sst2")

    df_train = data["train"].to_pandas().set_index("idx").rename(columns={"sentence": "text"})
    df_train = df_train[df_train["text"].str.len() > 0]
    df_test = data["validation"].to_pandas().set_index("idx").rename(columns={"sentence": "text"})
    df_test = df_test[df_test["text"].str.len() > 0]
    
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_idx = rs.permutation(df_all.index.to_numpy())
    train_list = all_idx[:len(df_train)]
    test_list = all_idx[len(df_train):]

    return df_all, train_list, test_list

def create_lists_agnews(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("ag_news")

    df_train = data["train"].add_column("idx", np.arange(len(data["train"]))).to_pandas().set_index("idx")
    df_train = df_train[df_train["text"].str.len() > 0]
    df_test = data["test"].add_column("idx", np.arange(len(data["test"]))).to_pandas().set_index("idx")
    df_test = df_test[df_test["text"].str.len() > 0]
    
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_idx = rs.permutation(df_all.index.to_numpy())
    train_list = all_idx[:len(df_train)]
    test_list = all_idx[len(df_train):]

    return df_all, train_list, test_list

def create_lists_dbpedia(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("fancyzhx/dbpedia_14")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    df_train = data["train"].to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"})
    df_train = df_train[df_train["text"].str.len() > 0]
    df_test = data["test"].to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"})
    df_test = df_test[df_test["text"].str.len() > 0]
    
    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_idx = rs.permutation(df_all.index.to_numpy())
    train_list = all_idx[:len(df_train)]
    test_list = all_idx[len(df_train):]

    return df_all, train_list, test_list


def create_lists_20newsgroups(seed):
    rs = np.random.RandomState(seed)
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

    df_train = data["train"].to_pandas().set_index("idx").drop(columns=["label_text"])
    df_train = df_train[df_train["text"].str.len() > 0]
    df_test = data["test"].to_pandas().set_index("idx").drop(columns=["label_text"])
    df_test = df_test[df_test["text"].str.len() > 0]

    df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    all_idx = rs.permutation(df_all.index.to_numpy())
    train_list = all_idx[:len(df_train)]
    test_list = all_idx[len(df_train):]

    return df_all, train_list, test_list

def create_lists_banking77(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("PolyAI/banking77",trust_remote_code=True)
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    all_data = datasets.concatenate_datasets([data["train"], data["test"]])

    df_all = all_data.to_pandas().set_index("idx")
    df_all = df_all[df_all["text"].str.len() > 0].reset_index(drop=True)
    all_idx = rs.permutation(df_all.index.to_numpy())
    train_list = all_idx[:len(data["train"])]
    test_list = all_idx[len(data["train"]):]

    return df_all, train_list, test_list


def main(dataset, lists_dir, data_dir, total_train_size=None, val_prop=None, repetitions=5, test_size=None, seed=0):

    if not Path(f"{data_dir}/all.csv").exists():

        if dataset == "sst2":
            df_all, train_list, test_list = create_lists_sst2(16237)
        elif dataset == "agnews":
            df_all, train_list, test_list = create_lists_agnews(782348)
        elif dataset == "dbpedia":
            df_all, train_list, test_list = create_lists_dbpedia(3399)
        elif dataset == "20newsgroups":
            df_all, train_list, test_list = create_lists_20newsgroups(8495)
        elif dataset == "banking77":
            df_all, train_list, test_list = create_lists_banking77(81234)
        else:
            raise ValueError("Invalid dataset")
        
        np.savetxt(f"{lists_dir}/train.txt", train_list, fmt="%d")
        np.savetxt(f"{lists_dir}/test.txt", test_list, fmt="%d")
        df_all.to_csv(f"{data_dir}/all.csv", index=True, header=True)
    
    if total_train_size is not None and val_prop is not None:
        train_list = np.loadtxt(f"{lists_dir}/train.txt", dtype=int)
        for i in range(repetitions):
            rs = np.random.RandomState(seed+i)
            idx = rs.permutation(train_list)
            val_size = int(total_train_size*val_prop)
            train_size = total_train_size - val_size
            train_idx = idx[:train_size]
            val_idx = idx[train_size:train_size+val_size]
            np.savetxt(f"{lists_dir}/train_{total_train_size}_{val_prop}_{i}.txt", train_idx, fmt="%s")
            np.savetxt(f"{lists_dir}/val_{total_train_size}_{val_prop}_{i}.txt", val_idx, fmt="%s")
    
    if test_size is not None:
        test_list = np.loadtxt(f"{lists_dir}/test.txt", dtype=int)
        rs = np.random.RandomState(seed)
        test_idx = rs.permutation(test_list)[:test_size]
        np.savetxt(f"{lists_dir}/test_{test_size}.txt", test_idx, fmt="%s")




if __name__ == "__main__":
    from fire import Fire
    Fire(main)