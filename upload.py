from huggingface_hub import login, HfApi
from huggingface_hub import create_repo
login(token="")

# Repository details
repo = "TP_NLP_best_validation_models_2"
create_repo(f"lestienne/{repo}", private=False)

# Initialize HfApi
api = HfApi()

# Folder to upload
folder = "/Users/juantollo/Library/Mobile Documents/com~apple~CloudDocs/Tesis/experimentos/output copy"

# Upload the folder with specified ignore patterns
api.upload_folder(
    folder_path=folder,
    path_in_repo="",
    repo_id=f"juantollo/{repo}",
    repo_type="model",  # Change to "model" if appropriate
    ignore_patterns=[
        "*/logs/.txt",  # Ignore all .txt files inside logs directories
        "*/_pycache_/",  # Ignore Python cache directories
        "*.tmp",  # Ignore temporary files
        "*.log",  # Ignore .log files
        "*/train_all",  # Ignore files named train_all (and variations like train_all.txt)
        "*/val_all",  # Ignore files named val_all (and variations like val_all.json)
    ],
)