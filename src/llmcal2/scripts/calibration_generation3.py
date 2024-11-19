


import pandas as pd
import numpy as np
import lightning as L
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint
import torch

from pathlib import Path

def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits

def apply_calibration(logits, alpha, beta, encoded_answers):
    logits = logits.type(torch.float32).log_softmax(dim=-1)
    pad_token_id = encoded_answers[0][-1]
    beta_norm = beta / encoded_answers.ne(pad_token_id).sum(dim=1).to(device=beta.device)
    expanded_logits = logits.view(-1,1) * alpha + beta_norm.view(1,-1)
    max_value = (expanded_logits == torch.max(expanded_logits)).nonzero()
    logits = expanded_logits[:,max_value[0,1]]
    return torch.log_softmax(logits, dim=0)

def sample(
    logits: torch.Tensor, alpha, beta, encoded_answers, temperature: float = 1.0, top_k = None, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1].log_softmax(dim=-1)
    logits = apply_calibration(logits, alpha, beta, encoded_answers)
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)


def next_token(model, input_pos: torch.Tensor, x: torch.Tensor, alpha, beta, encoded_answers, **kwargs) -> torch.Tensor:
    logits = model(x, input_pos)
    _next = sample(logits, alpha, beta, encoded_answers, **kwargs).to(dtype=torch.int64)
    return _next

def generate_fn(
    model,
    prompt: torch.Tensor,
    alpha, beta, encoded_answers,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k = None,
    top_p = 1.0,
    stop_tokens = (),
    include_prompt: bool,
    include_eos: bool,
):
    """
    Generates tokens for a single prompt.

    Args:
        model: The model to use.
        prompt: The tokenized prompt to generate from.
        max_returned_tokens: The maximum number of new tokens to return. Does not include the prompt tokens.
        temperature: The temp to pass to sample().
        top_k: The top_k to pass to sample().
        top_p: The top_p to pass to sample().
        stop_tokens: A tuple of stop sequences. If any of the sequences are generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops early.
    """



    prompt_size = prompt.size(0)
    device = prompt.device

    assert max_returned_tokens > prompt_size, f"Not enough space for {prompt_size} prompt tokens in a context length of {max_returned_tokens}."
    if model.max_seq_length < max_returned_tokens - 1:
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    # Yield the prompt if include_prompt is True
    if include_prompt:
        yield prompt

    stop_progress = [0] * len(stop_tokens)
    yielded_idx = 0

    # Generate output tokens.
    # The first token generated is the prefill token.
    # The input_pos for this token is the width of the entire prompt.
    # For subsequent iterations, it's the index in the context for the token that we're generating.
    tokens = []
    token = prompt
    prefill_token = True
    input_pos = torch.arange(0, prompt_size, device=device, dtype=torch.int64)
    for current_idx in range(max_returned_tokens - prompt_size):

        # Generate the token
        token = next_token(model, input_pos, token.view(1, -1), alpha, beta, encoded_answers, temperature=temperature, top_k=top_k, top_p=top_p)
        tokens.append(token)
        int_token = token.item()

        # Check for stop sequences
        # For each stop sequence, we keep a running total of how many are matched in stop_progress.
        # If the current token matches the next token in the stop sequence, we increment the
        # running total and hold off on yielding the token.
        for i, seq in enumerate(stop_tokens):
            if int_token == seq[stop_progress[i]]:
                stop_progress[i] += 1
                if stop_progress[i] == len(seq):
                    if include_eos:
                        yield from tokens[yielded_idx:]
                    return
            else:
                stop_progress[i] = 0

        # Yield tokens that are not part of a stop sequence in progress.
        # If there are no stop sequences, then that's all of them.
        if stop_tokens:
            safe_idx = len(tokens) - max(stop_progress)
        else:
            safe_idx = current_idx + 1 # include the token just generated

        if yielded_idx < safe_idx:
            y_tokens = tokens[yielded_idx : safe_idx]
            yield from y_tokens
            yielded_idx = safe_idx

        # Update input_pos for the next iteration.
        if prefill_token:
            prefill_token = False
            input_pos = torch.tensor([prompt_size], device=device, dtype=torch.int64)
        else:
            input_pos.add_(1)

    # Yield any remaining tokens
    if yielded_idx < len(tokens):
        yield from tokens[yielded_idx:]

    

@torch.inference_mode()
def cal_generate(model, encoded_answers, alpha, beta, prompt, max_returned_tokens, temperature=1.0, top_k=1, top_p=0.0, eos_id=None, include_prompt=True):

    token_list = list(generate_fn(
        alpha=alpha,
        beta=beta,
        encoded_answers=encoded_answers,
        include_prompt=include_prompt,
        include_eos=True,
        model=model,
        prompt=prompt,
        max_returned_tokens=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_tokens=(([eos_id],) if eos_id is not None else ())
    ))

    return torch.cat(token_list) if not len(token_list) == 0 else torch.Tensor()




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
    
    # model_path = "outputs/adaptation/llama3.2-1b/lora_fs/banking77_4928_0.3_0/checkpoint"
    # cal_path = "outputs/adaptation/llama3.2-1b/lora_fs_plus_dp_cal/banking77_4928_0.3_0"
    # peft = "lora"
    peft = None
    max_seq_length = 1024
    idx = 5748
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

    # Encode and pad answers
    encoded_answers = [tokenizer.encode(answer, device=fabric.device, bos=False, eos=False).tolist() for answer in answers]
    print(pd.Series(encoded_answers).str.len().to_csv("banking77.csv",header=False,index=False))
    # pad_token_id = encoded_answers[0][-1]
    # max_length = max(len(encoded_answer) for encoded_answer in encoded_answers)
    # padded_answers = [ans + [pad_token_id] * (max_length - len(ans)) for ans in encoded_answers]
    # encoded_answers = torch.tensor(padded_answers)
    
    # checkpoint = torch.load(cal_path / "last.ckpt", weights_only=False)
    # alpha = checkpoint["model"]["alpha"].to(fabric.device)
    # beta = checkpoint["model"]["beta"].to(fabric.device)
    
    # model = fabric.setup_module(model)
    # load_checkpoint(fabric, model, model_path / "lit_model.pth", strict=False)
    # if peft == "lora":
    #     from litgpt.lora import merge_lora_weights
    #     lora_checkpoint_path = model_path / "lit_model.pth.lora"
    #     load_checkpoint(fabric, model, lora_checkpoint_path, strict=False)
    #     merge_lora_weights(model)

    # # Generate
    # output = cal_generate(model, encoded_answers, alpha, beta, encoded, max_returned_tokens=max_returned_tokens, temperature=temperature, top_k=top_k, top_p=top_p, eos_id=tokenizer.eos_id)
    # fabric.print(tokenizer.decode(output))
    # fabric.print("\n\nCorrect answer:", answers[label])



if __name__ == "__main__":
    setup()