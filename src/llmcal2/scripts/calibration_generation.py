


import pandas as pd
import numpy as np
import lightning as L
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint
from litgpt.generate.base import generate
import torch

from pathlib import Path


class GPTWithCalParams(torch.nn.Module):

    def __init__(self, model, alpha, beta, encoded_answers):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.encoded_answers = encoded_answers
        self.max_seq_length = model.max_seq_length
    
    @property
    def padded_answers(self):
        pad_token_id = self.encoded_answers[0][-1]
        max_length = max(len(encoded_answer) for encoded_answer in self.encoded_answers)
        padded_answers = [ans + [pad_token_id] * (max_length - len(ans)) for ans in self.encoded_answers]
        return torch.tensor(padded_answers)
        
    def forward(self, *args, **kwargs):
        logprobs = self.model(*args, **kwargs).log_softmax(dim=2)
        padded_answers = self.padded_answers.to(logprobs.device)
        uniques = torch.unique(padded_answers)
        import pdb; pdb.set_trace()
        mask = torch.zeros(logprobs.size(2), device=logprobs.device, dtype=torch.bool)
        mask[uniques] = True
        logprobs.masked_fill_(~mask.unsqueeze(0).unsqueeze(0), -float("inf"))
        return logprobs



def setup():

    torch.set_float32_matmul_precision("high")
    fabric = L.Fabric(
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision="bf16-true",
    )
    

    # model_path = "outputs/checkpoints/meta-llama/Llama-3.2-1B"
    model_path = "outputs/checkpoints/meta-llama/Llama-3.2-1B-Instruct"
    cal_path = "outputs/adaptation/llama3.2-1b/instruct_plus_dp_cal/banking77_4928_0.3_0"
    # peft = "lora"
    peft = None
    max_seq_length = 1024
    idx = 0
    max_new_tokens = 40
    temperature = 1.0
    top_k = 1
    top_p = 0.0

    data = pd.read_json("outputs/prompts/llama3.2-1b/banking77/all.jsonl", lines=True, orient="records")
    prompt, answers, label = data.iloc[idx]["prompt"], data.iloc[idx]["answer"], data.iloc[idx]["label"]

    model_path = Path(model_path)
    cal_path = Path(cal_path)
    fabric.launch(main, model_path, cal_path, peft, max_seq_length, prompt, answers, label, max_new_tokens, temperature, top_k, top_p)



def main(fabric, model_path, cal_path, peft, max_seq_length, prompt, answers, label, max_new_tokens, temperature, top_k, top_p):

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
    config = Config.from_file(model_path / "model_config.yaml", **peft_kwargs)

    # Load tokenizer
    tokenizer = Tokenizer(model_path)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        model.max_seq_length = max_seq_length
        model.set_kv_cache(batch_size=1, max_seq_length=max_seq_length)

    checkpoint = torch.load(cal_path / "last.ckpt", weights_only=False)
    encoded_answers = [tokenizer.encode(answer, device=fabric.device, bos=False, eos=True).tolist() for answer in answers]
    alpha = checkpoint["model"]["alpha"].to(fabric.device, dtype=next(model.parameters()).dtype)
    beta = checkpoint["model"]["beta"].to(fabric.device, dtype=next(model.parameters()).dtype)
    model = GPTWithCalParams(model, alpha, beta, encoded_answers)
    model = fabric.setup_module(model)
    load_checkpoint(fabric, model, model_path / "lit_model.pth", strict=False)
    if peft == "lora":
        from litgpt.lora import merge_lora_weights
        lora_checkpoint_path = model_path / "lit_model.pth.lora"
        load_checkpoint(fabric, model, lora_checkpoint_path, strict=False)
        merge_lora_weights(model)

    # Generate
    output = generate(model, encoded, max_returned_tokens=max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id)
    fabric.print(tokenizer.decode(output))
    fabric.print("\n\nCorrect answer:", answers[label])



if __name__ == "__main__":
    setup()