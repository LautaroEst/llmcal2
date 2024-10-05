
import numpy as np
import datasets
datasets.disable_caching()

def create_lists_sst2(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("nyu-mll/glue", "sst2")

    df_train = data["train"].to_pandas().set_index("idx").rename(columns={"sentence": "text"})
    df_train = df_train[df_train["text"].str.len() > 0]
    train_list = rs.permutation(df_train.index.to_numpy())

    df_test = data["validation"].to_pandas().set_index("idx").rename(columns={"sentence": "text"})
    df_test = df_test[df_test["text"].str.len() > 0]
    test_list = rs.permutation(df_test.index.to_numpy())

    return train_list, test_list

def create_lists_agnews(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("ag_news")

    df_train = data["train"].add_column("idx", np.arange(len(data["train"]))).to_pandas().set_index("idx")
    df_train = df_train[df_train["text"].str.len() > 0]
    train_list = rs.permutation(df_train.index.to_numpy())

    df_test = data["test"].add_column("idx", np.arange(len(data["test"]))).to_pandas().set_index("idx")
    df_test = df_test[df_test["text"].str.len() > 0]
    test_list = rs.permutation(df_test.index.to_numpy())

    return train_list, test_list

def create_lists_dbpedia(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("fancyzhx/dbpedia_14")
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["test"])))

    df_train = data["train"].to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"})
    df_train = df_train[df_train["text"].str.len() > 0]
    train_list = rs.permutation(df_train.index.to_numpy())

    df_test = data["test"].to_pandas().set_index("idx").loc[:,["content","label"]].rename(columns={"content": "text"})
    df_test = df_test[df_test["text"].str.len() > 0]
    test_list = rs.permutation(df_test.index.to_numpy())[:7000]

    return train_list, test_list


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
    train_list = rs.permutation(df_train.index.to_numpy())

    df_test = data["test"].to_pandas().set_index("idx").drop(columns=["label_text"])
    df_test = df_test[df_test["text"].str.len() > 0]
    test_list = rs.permutation(df_test.index.to_numpy())

    return train_list, test_list

def create_lists_banking77(seed):
    rs = np.random.RandomState(seed)
    data = datasets.load_dataset("PolyAI/banking77",trust_remote_code=True)
    data["train"] = data["train"].add_column("idx", np.arange(len(data["train"])))
    data["test"] = data["test"].add_column("idx", np.arange(len(data["train"]), len(data["train"])+len(data["test"])))
    all_data = datasets.concatenate_datasets([data["train"], data["test"]])

    df_all = all_data.to_pandas().set_index("idx")
    df_all = df_all[df_all["text"].str.len() > 0]
    train_list = rs.permutation(df_all.iloc[len(data["test"]):].index.to_numpy())
    test_list = rs.permutation(df_all.iloc[:len(data["test"])].index.to_numpy())

    return train_list, test_list


def main(dataset, output_dir):

    if dataset == "sst2":
        train_list, test_list = create_lists_sst2(16237)
    elif dataset == "agnews":
        train_list, test_list = create_lists_agnews(782348)
    elif dataset == "dbpedia":
        train_list, test_list = create_lists_dbpedia(3399)
    elif dataset == "20newsgroups":
        train_list, test_list = create_lists_20newsgroups(8495)
    elif dataset == "banking77":
        train_list, test_list = create_lists_banking77(81234)
    else:
        raise ValueError("Invalid dataset")
    
    np.savetxt(f"{output_dir}/train--all.txt", train_list, fmt="%s")
    np.savetxt(f"{output_dir}/test--all.txt", test_list, fmt="%s")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)