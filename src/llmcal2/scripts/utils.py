import numpy as np
import torch
import os
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger as _CSVLogger
import pandas as pd
from ..utils import save_yaml


class GenerativeCollator:

    def __init__(self, pad_token_id, max_seq_len):
        # batch = {"idx": ..., "prompt_ids": ..., "answers_ids": ...}
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        prompts_ids = []
        prompt_masks = []
        answers_ids = []
        max_ans_len = max([max([ans.shape[0] for ans in sample["answers_ids"]]) for sample in batch])

        max_prompt_len = min(self.max_seq_len - max_ans_len, max([sample["prompt_ids"].shape[0] for sample in batch]))
        for sample in batch:
            seq = sample["prompt_ids"][-max_prompt_len:]
            prompts_ids.append(torch.cat([torch.ones(max_prompt_len - seq.shape[0], dtype=torch.long) * self.pad_token_id, seq]))
            prompt_masks.append(torch.cat([torch.zeros(max_prompt_len - seq.shape[0], dtype=torch.long), torch.ones(seq.shape[0], dtype=torch.long)]))
            answers_ids.append(sample["answers_ids"])
        return {
            "idx": torch.stack([sample["idx"] for sample in batch]),
            "prompt_ids": torch.stack(prompts_ids),
            "prompt_mask": torch.stack(prompt_masks),
            "answers_ids": answers_ids,
            "use_ids": torch.stack([sample["use_ids"] for sample in batch]),
            "label": torch.stack([sample["label"] for sample in batch])
        }
    


class TBLogger(TensorBoardLogger):

    def __init__(self, save_dir):
        _save_dir = "/".join(save_dir.split("/")[:-1])
        _version = save_dir.split("/")[-1]
        super().__init__(
            save_dir=_save_dir,
            name="",
            version=_version,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
        )

    def log_hyperparams(self, hyperparams, metrics = None):
        super().log_hyperparams(hyperparams, metrics)
        save_yaml(hyperparams, os.path.join(self.log_dir, "hyperparams.yaml"))


class CSVLogger(_CSVLogger):

    def __init__(self, save_dir):
        _save_dir = "/".join(save_dir.split("/")[:-1])
        _version = save_dir.split("/")[-1]
        super().__init__(
            save_dir=_save_dir,
            name="",
            version=_version,
            prefix="",
            flush_logs_every_n_steps=1,
        )
        if os.path.exists(os.path.join(self.log_dir, "metrics.csv")):
            self.experiment.metrics = pd.read_csv(os.path.join(self.log_dir, "metrics.csv")).to_dict(orient="records")
        

def load_data(data_dir, train_list, val_list, test_list):

    # Load data
    df_train = pd.read_json(data_dir / f"train_prompt.jsonl", lines=True).set_index("idx")
    df_test = pd.read_json(data_dir / f"test_prompt.jsonl", lines=True).set_index("idx")

    data = {}
    for split, l, df in zip(["train", "val", "test"], [train_list, val_list, test_list], [df_train, df_train, df_test]):
        if l is not None:
            l = np.loadtxt(l, dtype=int)
            data[split] = df.loc[l].copy()
    
    return data