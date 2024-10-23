
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Union, Literal
import warnings

from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    check_nvlink_connectivity,
    load_checkpoint
)
import lightning as L
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from tqdm import tqdm

from ..utils import get_dataloader


def setup(
    checkpoint_dir,
    data_path: str,
    output_dir: str,
    prediction_list: Optional[str] = None,
    peft: Union[Literal["lora", "adapter"], None] = None,
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    batch_size: int = 1,
    max_seq_length: int = 1024,
    **peft_kwargs,
):
    # Basic setup
    torch.set_float32_matmul_precision("high")
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load config file
    check_valid_checkpoint_dir(checkpoint_dir)
    if peft is None:
        from litgpt.config import Config
        from litgpt.model import Block
    elif peft == "lora":
        from litgpt.lora import Config, Block
    elif peft == "adapter":
        from litgpt.adapter import Config, Block
    else:
        raise ValueError(f"Unknown peft type: {peft}")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **peft_kwargs)

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
    )
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    # Launch
    fabric.launch(main, peft, config, checkpoint_dir, data_path, output_dir, prediction_list, batch_size, max_seq_length)


def main(
    fabric: L.Fabric,
    peft,
    config,
    checkpoint_dir,
    data_path,
    output_dir,
    prediction_list,
    batch_size,
    max_seq_length,
):
    
    # Seed everything
    fabric.seed_everything(92837)

    # Load model parameters from checkpoint
    if peft is None:
        from litgpt.model import GPT
    elif peft == "lora":
        from litgpt.lora import GPT
    elif peft == "adapter":
        from litgpt.adapter import GPT
    else:
        raise ValueError(f"Unknown peft type: {peft}")
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        model.max_seq_length = max_seq_length
        model.set_kv_cache(batch_size=batch_size, max_seq_length=max_seq_length)
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, checkpoint_path, strict=False)

    if peft == "lora":
        from litgpt.lora import merge_lora_weights
        lora_checkpoint_path = checkpoint_dir / "lit_model.pth.lora"
        load_checkpoint(fabric, model, lora_checkpoint_path, strict=False)
        merge_lora_weights(model)
    elif peft == "adapter":
        adapter_checkpoint_path = checkpoint_dir / "lit_model.pth.adapter"
        load_checkpoint(fabric, model, adapter_checkpoint_path, strict=False)

    # Load tokenizer
    tokenizer = Tokenizer(checkpoint_dir)

    # Predict
    dataloader = get_dataloader(data_path, prediction_list, tokenizer, batch_size, pad_token_id=0, max_seq_length=model.max_seq_length, shuffle = False)
    dataloader = fabric.setup_dataloaders(dataloader)
    predictions = predict(fabric, model, dataloader)
    if fabric.global_rank == 0:
        pd.DataFrame(predictions["logits"], index=predictions["idx"]).to_csv(output_dir / f"logits.csv", index=True, header=False)
        pd.DataFrame(predictions["label"], index=predictions["idx"]).to_csv(output_dir / f"labels.csv", index=True, header=False)


def predict_step(model, indices, prompt_ids, prompt_mask, answers_ids, labels):
    logits = []
    for input_ids, attention_mask, answers in zip(prompt_ids, prompt_mask, answers_ids):
        input_ids = input_ids[attention_mask == 1].unsqueeze(0)
        T = torch.sum(attention_mask)
        input_pos = torch.arange(0, T, device=input_ids.device, dtype=input_ids.dtype)
        output = model(idx=input_ids, input_pos=input_pos)
        answers_logits = []
        for answer in answers:
            answer = answer.unsqueeze(0)
            input_pos = torch.arange(T, answer.shape[1] + T, device=answer.device, dtype=answer.dtype) 
            ans_out = model(idx=answer, input_pos=input_pos)
            logprobs = torch.cat([output[:,-1:,:], ans_out[:,:-1,:]], dim=1).log_softmax(dim=2)
            index = answer.unsqueeze(2)
            gather_probs = torch.gather(logprobs, -1, index).squeeze(2)
            ans_logit = gather_probs.sum()
            answers_logits.append(ans_logit)
        logits.append(torch.stack(answers_logits, dim=0))
    logits = torch.stack(logits, dim=0)
    return {"idx": indices, "logits": logits, "label": labels}


def predict(fabric, model, dataloader):
    predict_outputs = {"idx": [], "logits": [], "label": []}
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = predict_step(model, batch["idx"], batch["prompt_ids"], batch["prompt_mask"], batch["answers_ids"], batch["label"])
            fabric.barrier()
            gathered_outputs = fabric.all_gather(outputs)
            if fabric.global_rank == 0:
                for k, v in gathered_outputs.items():
                    if k in ["idx", "label"]:
                        predict_outputs[k].append(v.cpu().long())
                    else:
                        predict_outputs[k].append(v.cpu().float())
    
    if fabric.global_rank == 0:
        for k, v in predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0).numpy()

    return predict_outputs
        

    


if __name__ == "__main__":
    from fire import Fire
    Fire(setup)