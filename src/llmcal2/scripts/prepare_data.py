
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils import load_yaml
from ..prompts import *

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path, index_col=0, header=0)

def load_shots(dataset, shots_list, answers):
    if shots_list is None:
        return []
    shots_list = np.loadtxt(shots_list, dtype=int)
    shots = dataset.loc[shots_list]
    shots["label"] = shots["label"].apply(lambda x: answers[x])
    return shots.to_dict(orient="records")


def select_prompt(model):
    if "llama3" in model:
        return Llama3Prompt
    elif "tinyllama" in model:
        return TinyLlamaPrompt
    elif "phi3" in model:
        return Phi3Prompt
    elif "pythia" in model:
        return PythiaPrompt
    else:
        raise ValueError(f"Unknown model: {model}")
    

def main(dataset_path, prompt_template, model, output_path, shots_list=None, max_characters=400):
    dataset_path = Path(dataset_path)
    prompt_template = Path(prompt_template)
    output_path = Path(output_path)

    # Load data
    dataset = load_dataset(dataset_path)
    
    # Create prompts
    prompt = load_yaml(prompt_template)
    prompt_template = prompt["prompt_template"]
    answers = prompt["answers_templates"]
    shots = load_shots(dataset, shots_list, answers)
    prompt_cls = select_prompt(model)
    prompt = prompt_cls(max_characters=max_characters)
    prompt.fit(prompt_template, shots)

    dataset["answer"] = [answers for _ in range(len(dataset))]
    dataset["prompt"] = prompt.apply(dataset["text"])
    dataset = dataset.loc[:,["prompt", "answer", "label"]].reset_index(drop=False).rename(columns={"index": "idx"})

    # Save prompts
    dataset.to_json(output_path, orient="records", lines=True)
    
    
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)
