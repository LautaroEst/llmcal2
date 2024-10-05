
import pandas as pd
import numpy as np


def main(data_dir, total_train_samples, val_prop, use_train_samples_as_val, random_state, output_dir):

    # Load data
    df_train = pd.read_json(f"{data_dir}/train_prompt.jsonl", lines=True).set_index("idx")
    
    # Split train into train and val
    train_samples = int(total_train_samples * (1 - val_prop))
    val_samples = total_train_samples - train_samples
    rs = np.random.RandomState(random_state)
    idx = rs.permutation(len(df_train))
    train_idx = idx[:train_samples]
    val_idx = idx[train_samples:train_samples + val_samples]
    df_val = df_train.iloc[val_idx]
    df_train = df_train.iloc[train_idx]

    if use_train_samples_as_val != -1:
        if val_prop != 0:
            raise ValueError("Cannot use both val_prop and use_train_samples_as_val")
        df_val = df_train.sample(min(use_train_samples_as_val,len(df_train)), random_state=random_state)

    np.savetxt(f"{output_dir}/train--total_train_samples={total_train_samples}_val_prop={val_prop:.1f}_random_state={random_state}.txt", df_train.index.to_numpy().astype(int), fmt='%d')
    np.savetxt(f"{output_dir}/val--total_train_samples={total_train_samples}_val_prop={val_prop:.1f}_random_state={random_state}.txt", df_val.index.to_numpy().astype(int), fmt='%d')





if __name__ == "__main__":
    from fire import Fire
    Fire(main)