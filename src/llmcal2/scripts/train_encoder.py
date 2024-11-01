
from pathlib import Path
import time
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from litgpt.utils import CycleIterator

from ..loggers import CSVLogger, TBLogger
from ..utils import save_yaml

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Experiment logs directory outputs*")

class JSONDataset(Dataset):
    def __init__(self, path, lst, tokenizer):
        self.lst = lst
        self.path = path
        self.tokenizer = tokenizer
        d = pd.read_csv(path, index_col=0, header=0).loc[lst]
        d = d.apply(self._transform, axis=1)
        self.data = d

    def _transform(self, sample):
        idx = torch.tensor(sample.name, dtype=torch.long)
        encoded_batch = self.tokenizer(sample["text"], return_tensors="pt", padding=False, truncation=True)
        input_ids = encoded_batch["input_ids"][0]
        attention_mask = encoded_batch["attention_mask"][0]
        label = torch.tensor(sample["label"], dtype=torch.long)
        return pd.Series({"idx": idx, "input_ids": input_ids, "attention_mask": attention_mask, "label": label})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()
    
    @property
    def num_classes(self):
        path = str(self.path)
        if "sst2" in path:
            return 2
        elif "agnews" in path:
            return 4
        elif "dbpedia" in path:
            return 14
        elif "20newsgroups" in path:
            return 20
        elif "banking77" in path:
            return 77
        else:
            raise ValueError(f"Unknown dataset: {path}")
        

class Collator:
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        max_prompt_len = min(self.max_length, max([sample["input_ids"].shape[0] for sample in batch]))
        for sample in batch:
            seq = sample["input_ids"][-max_prompt_len:]
            input_ids.append(torch.cat([seq, torch.ones(max_prompt_len - seq.shape[0], dtype=torch.long) * self.pad_token_id]))
            attention_mask.append(torch.cat([torch.ones(seq.shape[0], dtype=torch.long), torch.zeros(max_prompt_len - seq.shape[0], dtype=torch.long)]))
        return {
            "idx": torch.stack([sample["idx"] for sample in batch]),
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "label": torch.stack([sample["label"] for sample in batch])
        }


def setup(
    base_checkpoint_dir,
    data_path,
    train_list,
    val_list,
    test_list,
    output_dir,
    predictions_dir,
    output_checkpoint_dir,
    log_dir,
    precision,
    devices,
    num_nodes,
    batch_size,
    train_save_interval,
    val_check_interval,
    learning_rate,
    optimizer,
    weight_decay,
    patience,
    seed,
):
    torch.set_float32_matmul_precision("high")
    
    base_checkpoint_dir = Path(base_checkpoint_dir)
    data_path = Path(data_path)
    train_list = np.loadtxt(train_list, dtype=int)
    val_list = np.loadtxt(val_list, dtype=int)
    test_list = np.loadtxt(test_list, dtype=int)
    output_dir = Path(output_dir)
    predictions_dir = Path(predictions_dir)
    output_checkpoint_dir = Path(output_checkpoint_dir)
    
    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy="auto",
        precision=precision,
        loggers=[
            CSVLogger(log_dir),
            TBLogger(log_dir),
        ]
    )

    train_args = {
        "batch_size": batch_size,
        "val_check_interval": val_check_interval,
        "train_save_interval": train_save_interval,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "patience": patience,
    }
    fabric.launch(main, base_checkpoint_dir, data_path, train_list, val_list, test_list, output_dir, predictions_dir, output_checkpoint_dir, train_args, seed)


def main(
    fabric,
    base_checkpoint_dir,
    data_path,
    train_list,
    val_list,
    test_list,
    output_dir,
    predictions_dir,
    output_checkpoint_dir,
    train_args,
    seed,
):
    # Seed everything
    fabric.seed_everything(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint_dir, clean_up_tokenization_spaces=False)

    # Load train data
    train_dataset = JSONDataset(data_path, train_list, tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_args["batch_size"], 
        shuffle=True, 
        collate_fn=Collator(tokenizer, max_length=512),
        generator=torch.Generator().manual_seed(seed)
    )

    # Load val data
    val_dataset = JSONDataset(data_path, val_list, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_args["batch_size"],
        shuffle=False,
        collate_fn=Collator(tokenizer, max_length=512),
    )

    # Load test data
    test_dataset = JSONDataset(data_path, test_list, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_args["batch_size"],
        shuffle=False,
        collate_fn=Collator(tokenizer, max_length=512),
    )
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(base_checkpoint_dir, num_labels=train_dataset.num_classes, ignore_mismatched_sizes=True)
    model = fabric.setup_module(model, move_to_device=True)

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if train_args["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])
    elif train_args["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {train_args['optimizer']}")
    optimizer = fabric.setup_optimizers(optimizer)

    # Train
    train_labels = torch.stack(train_dataset.data["label"].to_list(), dim=0)
    train_priors = train_labels.bincount(minlength=train_dataset.num_classes).float() / len(train_dataset)
    train_prior_loss = -torch.log(train_priors[train_labels]).mean().item()
    val_labels = torch.stack(val_dataset.data["label"].to_list(), dim=0)
    val_priors = val_labels.bincount(minlength=val_dataset.num_classes).float() / len(val_dataset)
    val_prior_loss = -torch.log(val_priors[val_labels]).mean().item()
    fit(fabric, model, train_loader, train_prior_loss, val_loader, val_prior_loss, optimizer, output_dir, seed, train_args)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    fabric.print("Training finished.")
    if fabric.global_rank == 0:
        save_yaml(train_args, output_dir / "train_args.yaml")
    fabric.save(output_checkpoint_dir / "model.bin", model.state_dict())

    # Predict on test
    predictions, labels, index = predict(fabric, model, test_loader)

    # Save predictions
    fabric.print("Saving predictions...")
    if fabric.global_rank == 0:
        pd.DataFrame(predictions, index=index).to_csv(predictions_dir / "logits.csv", index=True, header=False)
        pd.DataFrame(labels, index=index).to_csv(predictions_dir / "labels.csv", index=True, header=False)


def fit(
    fabric,
    model,
    train_loader,
    train_prior_loss,
    val_loader,
    val_prior_loss,
    optimizer,
    output_dir,
    seed,
    train_args
):
    
    if (state_dict_path := output_dir / "last.ckpt").exists():
        state = torch.load(state_dict_path, weights_only=False)
        model.load_state_dict(state["model"], strict=False) # only lora params are saved
    else:
        state = {
            "model": model.state_dict(),
            "step_count": 0,
            "iter_num": 0,
            "train_loss": float("inf"),
            "best_val_loss": float("inf"),
            "last_val_loss": float("inf"),
            "patience_count": 0,
            "start_time": time.perf_counter(),
            "end_time": None,
        }

    train_iterator = CycleIterator(train_loader)
    rs = np.random.RandomState(seed)
    gradient_accumulation_iters = 1

    # Define the loss
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    # Advance until state
    step_count = 0
    iter_num = 0
    while step_count < state["step_count"]:
        iter_num += 1
        batch = next(train_iterator)
        is_accumulating = iter_num % gradient_accumulation_iters != 0
        if not is_accumulating:
            step_count += 1

    # Continue training
    model.train()
    while state["patience_count"] < train_args["patience"]:
        state["iter_num"] += 1
        batch = next(train_iterator)
        iter_t0 = time.perf_counter()

        # Perform forward and backward pass
        is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            output = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(output.logits, batch["label"])
            fabric.backward(loss)
        state["train_loss"] = loss.item() / train_prior_loss
        
        # Perform optimizer step
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        # Log train loss
        if not is_accumulating or state["iter_num"] == 1:
            t1 = time.perf_counter()
            metrics = {
                "train/loss": state["train_loss"],
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
            }
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" train loss: {metrics['train/loss']:.3f},"
                f" val loss: {state['last_val_loss']:.3f} |"
                f" best val loss: {state['best_val_loss']:.3f} |"
                f" patience: {state['patience_count']} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
            )
            fabric.log_dict(metrics, step=state["step_count"])

        # Validate
        if not is_accumulating and state["step_count"] % train_args["val_check_interval"] == 0:
            state["last_val_loss"] = validate(fabric, model, val_loader) / val_prior_loss
            fabric.log_dict({
                "val/loss": state["last_val_loss"],
            }, step=state["step_count"])
            if state["last_val_loss"] < state["best_val_loss"]:
                state.update({
                    "model": model.state_dict(),
                    "end_time": time.perf_counter(),
                    "best_val_loss": state["last_val_loss"],
                    "patience_count": 0,
                })
                fabric.save(output_dir / "best.ckpt", state)
            else:
                state["patience_count"] += 1
            fabric.barrier()

        # Save last checkpoint
        if not is_accumulating and state["step_count"] % train_args["train_save_interval"] == 0:
            state["model"] = model.state_dict()
            state["end_time"] = time.perf_counter()
            fabric.save(output_dir / "last.ckpt", state)


@torch.no_grad()
def validate(fabric, model, val_dataloader):
    total_loss = 0
    total_num_samples = 0
    model.eval()
    fabric.print("Validating...")
    for batch in val_dataloader:
        output = model(batch["input_ids"], batch["attention_mask"])
        loss = torch.nn.functional.cross_entropy(output.logits, batch["label"], reduction="sum")
        total_loss += loss.item()
        total_num_samples += len(batch["label"])
    model.train()
    return total_loss / total_num_samples

@torch.no_grad()
def predict(fabric, model, test_loader):
    model.eval()
    fabric.print("Predicting...")
    predictions = {"logits": [], "labels": [], "idx": []}
    for batch in test_loader:
        output = model(batch["input_ids"], batch["attention_mask"])
        predictions["logits"].append(output.logits.cpu().numpy())
        predictions["labels"].append(batch["label"].cpu().numpy())
        predictions["idx"].append(batch["idx"].cpu().numpy())
    logits = np.vstack(predictions["logits"])
    labels = np.hstack(predictions["labels"])
    index = np.hstack(predictions["idx"])
    return logits, labels, index

if __name__ == "__main__":
    from fire import Fire
    Fire(setup)