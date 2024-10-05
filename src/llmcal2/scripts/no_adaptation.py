
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from litgpt import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from litgpt import Config
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load

from ..models.generative_no_adaptation import Generative, GenerativeLanguageModel
from .utils import GenerativeCollator, load_data


class PromptsDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        return {
            "idx": torch.tensor(data.name, dtype=torch.long),
            "prompt_ids": data["prompt_ids"],
            "answers_ids": data["answers_ids"],
            "label": torch.tensor(data["label"], dtype=torch.long),
        }
    

def process_dataframe(df, tokenizer):

    def transform(sample):
        prompt_ids = tokenizer.encode(sample["prompt"], bos=True).long()
        answers_ids = [tokenizer.encode(ans, bos=True)[1:].long() for ans in sample["answer"]]
        return pd.Series({"prompt_ids": prompt_ids, "answers_ids": answers_ids, "label": sample["label"]})
    
    df = df.apply(transform, axis=1)
    dataset = PromptsDataset(df)
    return dataset


def main(
    data_dir,
    checkpoint_dir,
    train_list = None,
    val_list = None,
    test_list = None,
    batch_size = 32,
    accelerator = "cpu",
    strategy = "auto",
    devices = 1,
    num_nodes = 1,
    precision = 32,
    output_dir = None,
):

    torch.set_float32_matmul_precision("high")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load data
    data = load_data(data_dir, train_list, val_list, test_list)

    # Load tokenizer
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_checkpoint(checkpoint_dir)

    # Process data
    for split, df in data.items():
        print(f"Processing {split}...")
        data[split] = process_dataframe(df, tokenizer)

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = strategy,
        devices = devices,
        num_nodes = num_nodes,
        precision = precision,
        logger = False,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        deterministic = True,
        profiler=None,
        default_root_dir = output_dir,
    )

    # Init base model
    with trainer.init_module(empty_init=True):
        gpt = Generative(config)
        gpt.set_kv_cache(batch_size=batch_size)
    checkpoint = lazy_load(checkpoint_dir / "lit_model.pth")
    gpt.load_state_dict(checkpoint, strict=False)
    model = GenerativeLanguageModel(gpt)

    for split, dataset in data.items():
        print(f"Predicting {split}...")
        dataloader = DataLoader(
            dataset, 
            shuffle=False, 
            batch_size=batch_size, 
            num_workers=4, 
            collate_fn=GenerativeCollator(0, config.block_size)
        )
        trainer.predict(model, dataloaders=dataloader)
        for k, v in model.predict_outputs.items():
            if k == "idx":
                continue
            if k == "label":
                v = v.numpy().astype(int)
            else:
                v = v.numpy()
            d = pd.DataFrame(v, index=model.predict_outputs["idx"].numpy().astype(int))
            d.to_csv(output_dir / f"{split}_{k}.csv", index=True, header=False)


    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)