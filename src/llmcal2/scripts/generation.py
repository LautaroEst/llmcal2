import pandas as pd
import numpy as np
import lightning as L
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint
from litgpt.generate.base import generate
import torch

from pathlib import Path


def setup():

    torch.set_float32_matmul_precision("high")
    fabric = L.Fabric(
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision="bf16-true",
    )

    # model_path = "outputs/checkpoints/meta-llama/Llama-3.2-1B"
    # model_path = "outputs/checkpoints/meta-llama/Llama-3.2-1B-Instruct"
    model_path = "outputs/adaptation/llama3.2-1b/lora_ans/banking77_4928_0.3_0/checkpoint"
    peft = "lora"
    # peft = None
    max_seq_length = 1024
    idx = 0
    max_new_tokens = 40
    temperature = 1.0
    top_k = 1
    top_p = 0.0

    data = pd.read_json("outputs/prompts/llama3.2-1b/banking77/all.jsonl", lines=True, orient="records")
    prompt, answers, label = data.iloc[idx]["prompt"], data.iloc[idx]["answer"], data.iloc[idx]["label"]

    fabric.launch(main, model_path, peft, max_seq_length, prompt, answers, label, max_new_tokens, temperature, top_k, top_p)

def main(fabric, model_path, peft, max_seq_length, prompt, answers, label, max_new_tokens, temperature, top_k, top_p):

    if peft is None:
        from litgpt.config import Config
        from litgpt.model import GPT
        peft_kwargs = {}
    elif peft == "lora":
        from litgpt.lora import Config, GPT
        peft_kwargs = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_query": True,
            "lora_key": True,
            "lora_value": True,
            "lora_projection": True,
            "lora_mlp": True,
            "lora_head": True,
        }
    else:
        raise ValueError(f"Unknown peft type: {peft}")
    model_path = Path(model_path)
    config = Config.from_file(model_path / "model_config.yaml", **peft_kwargs)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        model.max_seq_length = max_seq_length
        model.set_kv_cache(batch_size=1, max_seq_length=max_seq_length)

    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, model_path / "lit_model.pth", strict=False)
    if peft == "lora":
        from litgpt.lora import merge_lora_weights
        lora_checkpoint_path = model_path / "lit_model.pth.lora"
        load_checkpoint(fabric, model, lora_checkpoint_path, strict=False)
        merge_lora_weights(model)

    # Load tokenizer
    tokenizer = Tokenizer(model_path)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens
    output = generate(model, encoded, max_returned_tokens=max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id)
    fabric.print(tokenizer.decode(output))
    fabric.print("\n\nCorrect answer:", answers[label])



if __name__ == "__main__":
    setup()