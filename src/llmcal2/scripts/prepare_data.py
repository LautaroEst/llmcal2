
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils import load_yaml

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path, index_col=0, header=0)

def select_prompt(prompt_template, model):
    if "llama3" in model:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{prompt_template}<|eot_id|>" # No newline
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{inpt}<|eot_id|>" # No newline
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "tinyllama" in model:
        return (
            "<|system|>\n"
            f"You are a friendly chatbot who always gives helpful, detailed, and polite answers. {prompt_template}</s>\n"
            "<|user|>\n"
            "{inpt}</s>\n"
            "<|assistant|>\n"
        )
    elif "phi3" in model:
        return (
            f'<|system|>\nYou are a helpful assistant. {prompt_template}'
            '<|end|>\n<|user|>\n{inpt}<|end|>\n<|assistant|>\n'
        )
    elif "pythia" in model:
        return (
            f'<|system|>\n{prompt_template}'
            '<|end|>\n<|user|>\n{inpt}<|end|>\n<|assistant|>\n'
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    

def main(dataset_path, prompt_template, model, output_path, max_characters=400):
    dataset_path = Path(dataset_path)
    prompt_template = Path(prompt_template)
    output_path = Path(output_path)

    # Load data
    dataset = load_dataset(dataset_path)
    
    # Create prompts
    prompt = load_yaml(prompt_template)
    prompt_template = prompt["prompt_template"]
    answers = prompt["answers_templates"]
    prompt = select_prompt(prompt_template, model)
    dataset["answer"] = [answers for _ in range(len(dataset))]
    dataset["prompt"] = dataset["text"].apply(lambda x: prompt.format(inpt=x[:max_characters]))
    dataset = dataset.loc[:,["prompt", "answer", "label"]].reset_index(drop=False).rename(columns={"index": "idx"})

    # Save prompts
    dataset.to_json(output_path, orient="records", lines=True)
    
    
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)
