
from functools import partial
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import yaml

from litgpt import Tokenizer

def load_yaml(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    if file is None:
        return {}
    return file

def save_yaml(data: dict, path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)

class JSONDataset(Dataset):

    def __init__(self, paths, lsts, tokenizer):
        self.lsts = lsts
        self.paths = paths
        self.tokenizer = tokenizer
        data = []
        for path, lst in zip(paths, lsts):
            d = pd.read_json(path, lines=True).set_index("idx").loc[lst].reset_index(drop=False)
            data.append(d.apply(self._transform, axis=1))
        self.data = pd.concat(data, ignore_index=False)

    def _transform(self, sample):
        idx = torch.tensor(sample["idx"], dtype=torch.long)
        prompt_ids = self.tokenizer.encode(sample["prompt"], bos=True).long()
        answers_ids = [self.tokenizer.encode(ans, bos=True)[1:].long() for ans in sample["answer"]]
        label = torch.tensor(sample["label"], dtype=torch.long)
        return pd.Series({"idx": idx, "prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": label})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()

class Collator:

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
            "label": torch.stack([sample["label"] for sample in batch])
        }


def get_dataloader(data_paths, lsts, tokenizer, batch_size = 1, pad_token_id = 0, max_seq_length = 2048, shuffle = False, seed = 42):
    dataset = JSONDataset(data_paths, lsts, tokenizer)
    collate_fn = Collator(pad_token_id=pad_token_id, max_seq_len=max_seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed)
    )
    return dataloader