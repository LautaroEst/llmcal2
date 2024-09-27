import yaml

def load_yaml(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    if file is None:
        return {}
    return file

def save_yaml(data: dict, path) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f)
