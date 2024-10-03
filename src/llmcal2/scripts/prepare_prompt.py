
import pandas as pd
from pathlib import Path
from ..utils import load_yaml

def load_data(data_dir):
    df_train = pd.read_csv(data_dir / f"train.csv", index_col=0, header=0)
    df_test = pd.read_csv(data_dir / f"test.csv", index_col=0, header=0)
    return {
        "train": df_train,
        "test": df_test
    }

def create_prompt_for_generative_model(df, config, max_words=400):
    df = df.copy()
    df["answer"] = [config["answers_templates"] for _ in range(len(df))]
    df["prompt"] = df["text"].apply(lambda x: config["prompt_template"].format(text=" ".join(x.split(" ")[:max_words])))
    df = df.loc[:,["prompt", "answer", "label"]]
    return df


def main(
    data_dir,
    prompt_template,
    output_dir,
    max_words = 400,
):
    # Load data
    data_dir = Path(data_dir)
    data = load_data(data_dir)

    # Create prompts
    prompt_template_config = load_yaml(prompt_template)
    template_type = prompt_template_config.pop("template_type")
    if template_type == "generative":
        for key in data:
            data[key] = create_prompt_for_generative_model(data[key], prompt_template_config, max_words=max_words)
    else:
        raise ValueError(f"Invalid template_type: {template_type}")

    # Save prompts
    output_dir = Path(output_dir)
    for key in data:
        data[key].reset_index().to_json(output_dir / f"{key}_prompt.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)