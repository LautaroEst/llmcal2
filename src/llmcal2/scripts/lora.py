
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from litgpt import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir
from litgpt.lora import Config, mark_only_lora_as_trainable, lora_filter, merge_lora_weights
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus

from ..models.lora import GenerativeLoRA, GenerativeLMLoRA, init_lora_linear_modules
from ..models.lora_norm import GenerativeLMLoRANorm
from .utils import GenerativeCollator, TBLogger, CSVLogger, load_data

class LMDataset(Dataset):

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
            "use_ids": data["use_ids"],
            "label": torch.tensor(data["label"], dtype=torch.long),
        }
    
def process_dataframe_for_train(df, tokenizer, include_answers=False, num_answers=None, priors=None, random_state=0):

    if include_answers:
        if num_answers is not None:
            rs = np.random.RandomState(random_state)
        def transform(sample):
            prompt_ids = tokenizer.encode(sample['prompt'], bos=True).long()
            answers_ids = [tokenizer.encode(ans, bos=True)[1:].long() for ans in sample["answer"]]
            use_ids = np.hstack((rs.choice([i for i in range(len(answers_ids)) if i != sample["label"]], num_answers, replace=False, p=priors),[sample["label"]])) if num_answers is not None and num_answers < len(answers_ids) - 1 else np.arange(len(answers_ids))
            use_ids = torch.from_numpy(use_ids).long()
            return pd.Series({"prompt_ids": prompt_ids, "answers_ids": answers_ids, "use_ids": use_ids, "label": sample["label"]})
    else:
        def transform(sample):
            prompt_with_answer = f"{sample['prompt']} {sample['answer'][sample['label']]}"
            prompt_ids = tokenizer.encode(prompt_with_answer, bos=True).long()
            answers_ids = [torch.tensor([],dtype=torch.long) for _ in sample["answer"]]
            use_ids = torch.tensor([],dtype=torch.long)
            return pd.Series({"prompt_ids": prompt_ids, "answers_ids": answers_ids, "use_ids": use_ids, "label": sample["label"]})
    
    df = df.apply(transform, axis=1)
    dataset = LMDataset(df)
    return dataset


def process_dataframe_for_predict(df, tokenizer):
    
    def transform(sample):
        prompt_ids = tokenizer.encode(sample['prompt'], bos=True).long()
        answers_ids = [tokenizer.encode(ans, bos=True)[1:].long() for ans in sample["answer"]]
        use_ids = torch.tensor([],dtype=torch.long)
        return pd.Series({"prompt_ids": prompt_ids, "answers_ids": answers_ids, "use_ids": use_ids, "label": sample["label"]})
    
    df = df.apply(transform, axis=1)
    dataset = LMDataset(df)
    return dataset




def main(
    data_dir,
    train_list,
    val_list,
    test_list,
    predict_on_val = False,
    random_state = 0,
    checkpoint_dir = None,
    norm = False,
    approx = None,
    dp_params = None,
    batch_size = 32,
    accelerator = "cpu",
    strategy = "auto",
    devices = 1,
    num_nodes = 1,
    precision = 32,
    max_epochs = 1000,
    max_steps = -1,
    val_check_interval = None,
    accumulate_grad_batches = 1,
    use_lora_checkpoint = False,
    lora_r = 1,
    lora_alpha = 0.5,
    lora_dropout = 0.1,
    lora_query = True,
    lora_key = True,
    lora_value = True,
    lora_projection = True,
    lora_mlp = True,
    lora_head = True,
    optimizer = "adamw",
    learning_rate = 1e-4,
    weight_decay = 0.0,
    output_dir = "output",
    log_dir = "output/logs",
    output_checkpoint_dir = "output/checkpoint",
):
    L.seed_everything(random_state)
    torch.set_float32_matmul_precision("high")
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_checkpoint_dir = Path(output_checkpoint_dir)

    # Load data
    data = load_data(data_dir, train_list, val_list, test_list)

    # Load tokenizer
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_checkpoint(
        checkpoint_dir,
        lora_r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        lora_query = lora_query,
        lora_key = lora_key,
        lora_value = lora_value,
        lora_projection = lora_projection,
        lora_mlp = lora_mlp,
        lora_head = lora_head,
    )

    # Process data
    train_data = {}
    train_priors = data["train"]["label"].value_counts(normalize=True).sort_index().values
    for split in ["train", "val"]:
        print(f"Processing {split}...")
        train_data[split] = process_dataframe_for_train(
            data[split], 
            tokenizer, 
            include_answers=norm, 
            num_answers=approx,
            priors=train_priors,
            random_state=random_state,
        )
        train_data[split] = DataLoader(
            train_data[split], 
            shuffle=split == "train",
            batch_size=batch_size, 
            num_workers=4, 
            collate_fn=GenerativeCollator(0, config.block_size)
        )

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = strategy,
        devices = devices,
        num_nodes = num_nodes,
        precision = precision,
        logger = [
            TBLogger(save_dir=log_dir),
            CSVLogger(save_dir=log_dir),
        ],
        max_epochs = max_epochs,
        max_steps = max_steps,
        val_check_interval = min(val_check_interval,len(train_data["train"])) if val_check_interval is not None else None,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        accumulate_grad_batches = accumulate_grad_batches,
        deterministic = True,
        profiler = None,
        default_root_dir = output_dir,
    )

    # Init base model
    with trainer.init_module(empty_init=True):
        gpt = GenerativeLoRA(config)
    mark_only_lora_as_trainable(gpt)
    
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint_path_lora = checkpoint_dir / "lit_model.pth.lora"  if use_lora_checkpoint else None
    checkpoint = lazy_load(checkpoint_path)
    gpt.load_state_dict(checkpoint, strict=False)
    if checkpoint_path_lora is not None:
        checkpoint = lazy_load(checkpoint_path_lora)
        gpt.load_state_dict(checkpoint, strict=False)
    else:
        init_lora_linear_modules(gpt)

    if norm:
        model = GenerativeLMLoRANorm(
            gpt = gpt,
            optimizer = optimizer,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
            dp_params = dp_params, 
            num_classes = len(data["train"].iloc[0]["answer"]),
        )
    else:
        model = GenerativeLMLoRA(
            gpt = gpt,
            optimizer = optimizer,
            learning_rate = learning_rate,
            weight_decay = weight_decay,
        )

    # -------------------
    # Fit the model
    # -------------------
    last_checkpoint_path = output_dir / "last.ckpt" if (output_dir / "last.ckpt").exists() else None
    trainer.fit(model, train_dataloaders=train_data["train"], val_dataloaders=train_data["val"], ckpt_path=last_checkpoint_path)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        print("Training interrupted.")
        return
    if train_data["val"] is not None:
        trainer.validate(model, dataloaders=train_data["val"])

    # Save best checkpoint and config files
    best_checkpoint_path = output_dir / "best.ckpt"
    if not best_checkpoint_path.exists():
        best_checkpoint_path = output_dir / "last.ckpt"
    checkpoint = lazy_load(best_checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    torch.save({k: v for k, v in model.gpt.state_dict().items() if lora_filter(k,v)}, output_checkpoint_dir / "lit_model.pth.lora")
    
    # Merge LoRA weights for inference
    merge_lora_weights(model.gpt)
    with trainer.init_module():
        model.gpt.set_kv_cache(batch_size=batch_size)


    # -----------------
    # Evaluate the model
    # -----------------
    for split in data:
        if (output_dir / f"{split}_logits.csv").exists():
            continue
        if split == "val" and not predict_on_val:
            continue

        print(f"Predicting {split}...")
        data[split] = process_dataframe_for_predict(data[split], tokenizer)
        dataloader = DataLoader(
            data[split], 
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
            if "embeddings" in k:
                d.to_pickle(Path(output_dir) / f"{split}_{k}.pkl")
            else:
                d.to_csv(output_dir / f"{split}_{k}.csv", index=True, header=False)


    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)