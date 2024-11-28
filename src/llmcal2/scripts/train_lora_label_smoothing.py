import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Literal, Optional, Union
import warnings

from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    check_nvlink_connectivity,
    load_checkpoint,
    CycleIterator,
)
from litgpt.lora import Config, Block, GPT, mark_only_lora_as_trainable, lora_filter
import lightning as L
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from tqdm import tqdm

from ..utils import get_dataloader, save_yaml
from ..loggers import TBLogger, CSVLogger

warnings.filterwarnings("ignore", category=UserWarning, message=".*Experiment logs directory outputs*")

def setup(
    base_checkpoint_dir: str,
    lora_checkpoint_dir: Optional[str] = None,
    data_paths: str = None,
    train_lists: str = None,
    train_logits: str = None,
    train_labels: str = None,
    val_lists: str = None,
    output_dir: str = None,
    output_checkpoint_dir: str = None,
    log_dir: str = None,
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    global_batch_size: int = 16,
    micro_batch_size: int = 1,
    val_check_interval = 16,
    learning_rate = 0.0001,
    optimizer = Literal["sgd", "adamw"],
    weight_decay = 0.0,
    loss: Literal["fs", "ans", "norm"] = "fs",
    patience: int = 10,
    max_steps: int = -1,
    seed = 0,
    max_seq_length: int = 1024,
    **lora_kwargs,
):
    
    # Basic setup
    torch.set_float32_matmul_precision("high")
    data_paths = [Path(data_path) for data_path in data_paths.split(",")]
    output_dir = Path(output_dir)
    output_checkpoint_dir = Path(output_checkpoint_dir)
    base_checkpoint_dir = Path(base_checkpoint_dir)
    lora_checkpoint_dir = Path(lora_checkpoint_dir) if lora_checkpoint_dir is not None else None
    train_lists = [np.loadtxt(train_list, dtype=int) for train_list in train_lists.split(",")]
    train_logits = pd.read_csv(train_logits, header=None, index_col=0)
    train_labels = pd.read_csv(train_labels, header=None, index_col=0)

    if val_lists is None:
        rs = np.random.RandomState(seed)
        val_lists = [rs.choice(train_list, min(len(train_list) // 10, 10), replace=False) for train_list in train_lists]
    else:
        val_lists = [np.loadtxt(val_list, dtype=int) for val_list in val_lists.split(",")]

    # Load config file
    check_valid_checkpoint_dir(base_checkpoint_dir)
    config = Config.from_file(base_checkpoint_dir / "model_config.yaml", **lora_kwargs)

    # Precision and quantization
    precision = precision or get_default_supported_precision(training=True)
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices * num_nodes > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    # Init fabric
    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        plugins=plugins,
        loggers=[
            TBLogger(save_dir=log_dir),
            CSVLogger(save_dir=log_dir),
        ]
    )
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    # Launch
    train_args = {
        "loss": "norm" if loss.startswith("norm-") else loss,
        "K": int(loss.split("-")[-1]) if loss.startswith("norm-") else None,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "val_check_interval": val_check_interval,
        "learning_rate": learning_rate,
        "optimizer_name": optimizer,
        "weight_decay": weight_decay,
        "patience": patience,
        "max_steps": max_steps,
    }
    fabric.launch(main, config, base_checkpoint_dir, lora_checkpoint_dir, data_paths, output_dir, output_checkpoint_dir, train_lists, train_logits, train_labels, val_lists, train_args, devices, seed, max_seq_length)


def main(
    fabric: L.Fabric,
    config: Config,
    base_checkpoint_dir: Path,
    lora_checkpoint_dir: Optional[Path],
    data_paths: Path,
    output_dir: Path,
    output_checkpoint_dir: Path,
    train_lists: np.ndarray,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    val_lists: np.ndarray,
    train_args: dict,
    devices: int,
    seed: int,
    max_seq_length: int,
):
    fabric.seed_everything(seed)

    # Init dataloaders
    tokenizer = Tokenizer(base_checkpoint_dir)
    train_dataloader = get_dataloader(data_paths, train_lists, tokenizer, train_args["micro_batch_size"], 0, max_seq_length, shuffle = True, seed = seed)
    val_dataloader = get_dataloader(data_paths, val_lists, tokenizer, train_args["micro_batch_size"], 0, max_seq_length, shuffle = False, seed = seed)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # Init Base model
    base_checkpoint_path = base_checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        model.max_seq_length = max_seq_length
    mark_only_lora_as_trainable(model)
    model = fabric.setup_module(model)

    # Init optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if train_args["optimizer_name"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])
    elif train_args["optimizer_name"] == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=train_args["learning_rate"], weight_decay=train_args["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {train_args['optimizer_name']}")
    optimizer = fabric.setup_optimizers(optimizer)

    # Load weights
    load_checkpoint(fabric, model, base_checkpoint_path, strict=False)
    if lora_checkpoint_dir is not None:
        lora_checkpoint_path = lora_checkpoint_dir / "lit_model.pth.lora"
        load_checkpoint(fabric, model, lora_checkpoint_path, strict=False)

    # Train
    fit(fabric, model, optimizer, train_dataloader, train_logits, train_labels, val_dataloader, devices, output_dir, seed, **train_args)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    fabric.print("Training finished.")

    if fabric.global_rank == 0:
        save_yaml(train_args, output_dir / "train_args.yaml")
    
    fabric.save(output_checkpoint_dir / "lit_model.pth.lora", {k: v for k, v in model.state_dict().items() if lora_filter(k,v)})
            


def fit(fabric, model, optimizer, train_dataloader, train_logits, train_labels, val_dataloader, devices, output_dir, seed, **train_args):
    
    if (state_dict_path := output_dir / "last.ckpt").exists():
        state = lazy_load(state_dict_path)
        model.load_state_dict(state["model"], strict=False) # only lora params are saved
    else:
        state = {
            "model": {k: v for k, v in model.state_dict().items() if lora_filter(k,v)},
            "step_count": 0,
            "iter_num": 0,
            "best_val_loss": float("inf"),
            "last_val_loss": float("inf"),
            "patience_count": 0,
            "cum_train_loss": 0,
            "cum_train_num_tokens": 0,
            "start_time": time.perf_counter(),
            "end_time": None,
        }

    train_iterator = CycleIterator(train_dataloader)
    rs = np.random.RandomState(seed)
    gradient_accumulation_iters = (train_args["global_batch_size"] // devices) // train_args["micro_batch_size"]

    # Define the loss
    if train_args["loss"] == "fs":
        loss_fn = FullSentenceLoss()
    elif train_args["loss"] == "ans":
        loss_fn = LossOnAnswer()
    elif train_args["loss"] == "norm":
        loss_fn = LossNormByAnswers(len(train_dataloader.dataset[0]["answers_ids"]), train_args["K"], seed)
    elif train_args["loss"] == "label_smooth":
        loss_fn = LossLabelSmooth(train_logits, train_labels)
    else:
        raise ValueError(f"Unknown loss: {train_args['loss']}")

    # Advance until state
    step_count = 0
    iter_num = 0
    while step_count < state["step_count"]:
        iter_num += 1
        batch = next(train_iterator)
        if train_args["loss"] == "norm":
            loss_fn.use_ids(batch["label"])
        is_accumulating = iter_num % gradient_accumulation_iters != 0
        if not is_accumulating:
            step_count += 1

    # Continue training
    model.train()
    stop_training = False
    while not stop_training:
        state["iter_num"] += 1
        batch = next(train_iterator)
        iter_t0 = time.perf_counter()

        # Perform forward and backward pass
        is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss, num_tokens = loss_fn(model, batch["idx"], batch["prompt_ids"], batch["prompt_mask"], batch["answers_ids"], batch["label"])
            fabric.backward(loss / num_tokens / gradient_accumulation_iters)
        
        # Accumulate loss for logging
        state["cum_train_loss"] += loss.item()
        state["cum_train_num_tokens"] += num_tokens

        # Perform optimizer step
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        # Log train loss
        if not is_accumulating or state["iter_num"] == 1:
            t1 = time.perf_counter()
            metrics = {
                "train/loss": state["cum_train_loss"] / state["cum_train_num_tokens"],
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
            val_loss, val_num_tokens = validate(fabric, model, val_dataloader, train_args, seed)
            state["last_val_loss"] = val_loss.item() / val_num_tokens
            fabric.log_dict({
                "val/loss": state["last_val_loss"],
            }, step=state["step_count"])
            if state["last_val_loss"] < state["best_val_loss"]:
                state.update({
                    "model": {k: v for k, v in model.state_dict().items() if lora_filter(k,v)},
                    "end_time": time.perf_counter(),
                    "best_val_loss": state["last_val_loss"],
                    "patience_count": 0,
                })
                fabric.save(output_dir / "best.ckpt", state)
            else:
                state["patience_count"] += 1
            fabric.barrier()

        # Save last checkpoint
        if not is_accumulating:
            state["model"] = {k: v for k, v in model.state_dict().items() if lora_filter(k,v)}
            state["end_time"] = time.perf_counter()
            fabric.save(output_dir / "last.ckpt", state)
            state["cum_train_loss"] = 0
            state["cum_train_num_tokens"] = 0

        # Check if training should stop
        if train_args["max_steps"] > 0:
            stop_training = state["step_count"] >= train_args["max_steps"]
        else:
            stop_training = state["patience_count"] >= train_args["patience"]


@torch.no_grad()
def validate(fabric, model, val_dataloader, train_args, seed):
    if train_args["loss"] == "fs":
        loss_fn = FullSentenceLoss()
    elif train_args["loss"] == "ans":
        loss_fn = LossOnAnswer()
    elif train_args["loss"] == "norm":
        loss_fn = LossNormByAnswers(len(val_dataloader.dataset[0]["answers_ids"]), train_args["K"], seed)

    total_loss = 0
    total_num_tokens = 0
    model.eval()
    fabric.print("Validating...")
    for batch in val_dataloader:
        loss, num_tokens = loss_fn(model, batch["prompt_ids"], batch["prompt_mask"], batch["answers_ids"], batch["label"])
        total_loss += loss
        total_num_tokens += num_tokens
    model.train()
    return total_loss, total_num_tokens

class LossLabelSmooth(torch.nn.Module):
    def __init__(self, train_logits, train_labels):
        super().__init__()
        self.train_logits = train_logits
        self.train_labels = train_labels

    def forward(self, model, indices, prompt_ids, prompt_mask, answers_ids, labels):
        loss = 0
        num_tokens = 0
        for idx, input_ids, attention_mask, answers, label in zip(indices, prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            class_logprobs = []
            for ans_ids in answers:
                full_input_ids = torch.cat([input_ids, ans_ids.unsqueeze(0)], dim=1)
                logprobs = model(full_input_ids, None)[:,input_ids.shape[1]-1:-1,:].log_softmax(dim=2)
                index = full_input_ids[:,input_ids.shape[1]:].unsqueeze(2)
                gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
                logprob = gather_logprobs.sum()
                class_logprobs.append(logprob) 
            logprobs = torch.log_softmax(torch.stack(class_logprobs, dim=0))
            ground_truth_logprobs = torch.log_softmax(torch.from_numpy(self.train_logits.loc[idx].values).to(logprobs.device))
            num_tokens = num_tokens + answers[label.item()].size(0)
            loss = loss + torch.nn.functional.kl_div(logprobs.unsqueeze(0), ground_truth_logprobs.unsqueeze(0), reduction="sum")
        return loss, num_tokens
        
class FullSentenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, prompt_ids, prompt_mask, answers_ids, labels):
        loss = 0
        num_tokens = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            full_input_ids = torch.cat([input_ids, answers[label.item()].unsqueeze(0)], dim=1)
            logprobs = model(full_input_ids, None)[:,:-1,:].log_softmax(dim=2)
            index = full_input_ids[:,1:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.size(1)
        return loss, num_tokens

class LossOnAnswer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, prompt_ids, prompt_mask, answers_ids, labels):
        loss = 0
        num_tokens = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            full_input_ids = torch.cat([input_ids, answers[label.item()].unsqueeze(0)], dim=1)
            logprobs = model(full_input_ids, None)[:,input_ids.shape[1]-1:-1,:].log_softmax(dim=2)
            index = full_input_ids[:,input_ids.shape[1]:].unsqueeze(2)
            gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
            loss = loss - gather_logprobs.sum()
            num_tokens = num_tokens + index.size(1)
        return loss, num_tokens


class LossNormByAnswers(torch.nn.Module):
    
    def __init__(self, total_num_answers, K = 5, seed = None):
        super().__init__()
        self.total_num_answers = total_num_answers
        self.K = K
        self._rs = np.random.RandomState(seed)

    def use_ids(self, label):
        if self.K is not None:
            use_ids = np.hstack(
                (self._rs.choice([i for i in range(self.total_num_answers) if i != label], min(self.K-1,self.total_num_answers-1), replace=False),[label.item()])
            )
        else:
            use_ids = np.arange(self.total_num_answers)
        return use_ids


    def forward(self, model, prompt_ids, prompt_mask, answers_ids, labels):
        loss = 0
        num_tokens = 0
        for input_ids, attention_mask, answers, label in zip(prompt_ids, prompt_mask, answers_ids, labels):
            input_ids = input_ids[attention_mask == 1].unsqueeze(0)
            class_logprobs = []
            use_ids = self.use_ids(label)
            for i, ans_ids in enumerate(answers):
                if i not in use_ids:
                    class_logprobs.append(torch.tensor(-float("inf"), device=input_ids.device))
                    continue
                full_input_ids = torch.cat([input_ids, ans_ids.unsqueeze(0)], dim=1)
                logprobs = model(full_input_ids, None)[:,input_ids.shape[1]-1:-1,:].log_softmax(dim=2)
                index = full_input_ids[:,input_ids.shape[1]:].unsqueeze(2)
                gather_logprobs = torch.gather(logprobs, -1, index).squeeze(2)
                logprob = gather_logprobs.sum()
                class_logprobs.append(logprob) 
            logits = torch.stack(class_logprobs, dim=0)
            num_tokens = num_tokens + answers[label.item()].size(0)
            loss = loss + torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0), reduction="sum")
        return loss, num_tokens


if __name__ == '__main__':
    from fire import Fire
    Fire(setup)