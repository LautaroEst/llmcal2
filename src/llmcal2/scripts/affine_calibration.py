

from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from .utils import TBLogger, CSVLogger
from ..models.affine_calibration import AffineCalibration

from litgpt.utils import check_valid_checkpoint_dir as check_valid_checkpoint_dir
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.pytorch.trainer.states import TrainerStatus

AFFINE_METHODS = ["dp_calibration", "temp_scaling", "bias_only"]


class CalibrationDataset(Dataset):

    def __init__(self, df_logits, df_labels):
        self.logits = torch.from_numpy(df_logits.values).float()
        self.labels = torch.from_numpy(df_labels.values).long().squeeze()
        self.index = torch.from_numpy(df_logits.index.values).long()

    def __len__(self):
        return self.logits.shape[0]
    
    def __getitem__(self, idx):
        return {
            "idx": self.index[idx],
            "logits": self.logits[idx],
            "label": self.labels[idx],
        }


def create_dataloaders(train_logits, train_labels, val_logits, val_labels, test_logits, test_labels, val_prop=0, random_state=42):
    data = {}
    is_train = train_logits is not None and train_labels is not None
    is_val = val_logits is not None and val_labels is not None
    is_test = test_logits is not None and test_labels is not None
    if is_train and is_val:
        df_train_logits = pd.read_csv(train_logits, index_col=0, header=None)
        df_train_labels = pd.read_csv(train_labels, index_col=0, header=None)
        df_val_logits = pd.read_csv(val_logits, index_col=0, header=None)
        df_val_labels = pd.read_csv(val_labels, index_col=0, header=None)
    elif is_train and not is_val:
        df_train_logits = pd.read_csv(train_logits, index_col=0, header=None)
        df_train_labels = pd.read_csv(train_labels, index_col=0, header=None)
        if val_prop == 0:
            df_val_logits = None
            df_val_labels = None
        else:
            total_train_samples = len(df_train_logits)
            train_samples = int(total_train_samples * (1 - val_prop))
            val_samples = total_train_samples - train_samples
            rs = np.random.RandomState(random_state+3)
            idx = rs.permutation(len(df_train_logits))
            train_idx = idx[:train_samples]
            val_idx = idx[train_samples:train_samples + val_samples]
            df_val_logits = df_train_logits.iloc[val_idx]
            df_val_labels = df_train_labels.iloc[val_idx]
            df_train_logits = df_train_logits.iloc[train_idx]
            df_train_labels = df_train_labels.iloc[train_idx]
    else:
        raise ValueError("Invalid input data.")

    dataset = CalibrationDataset(df_train_logits, df_train_labels)
    data["train"] = DataLoader(
        dataset, 
        batch_size=len(dataset), 
        shuffle=True, 
        num_workers=4, 
    )
    if df_val_logits is not None:
        dataset = CalibrationDataset(df_val_logits, df_val_labels)
        data["val"] = DataLoader(
            dataset, 
            batch_size=len(dataset), 
            shuffle=False, 
            num_workers=4, 
        )
    else:
        data["val"] = None

    if is_test:
        df_test_logits = pd.read_csv(test_logits, index_col=0, header=None)
        df_test_labels = pd.read_csv(test_labels, index_col=0, header=None)
        dataset = CalibrationDataset(df_test_logits, df_test_labels)
        data["test"] = DataLoader(
            dataset, 
            batch_size=len(dataset), 
            shuffle=False, 
            num_workers=4, 
        )
    return data

def main(
    train_logits = None,
    train_labels = None,
    val_logits = None,
    val_labels = None,
    test_logits = None,
    test_labels = None,
    val_prop = 0.1,
    random_state = 42,
    method = "dp_calibration",
    max_ls = 40,
    learning_rate = 0.001,
    accelerator = "cpu",
    max_epochs = 1000,
    output_dir = "output",
    checkpoint_dir = "output/checkpoint",
    log_dir = "output/logs",
):

    L.seed_everything(random_state)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load dataset
    data = create_dataloaders(train_logits, train_labels, val_logits, val_labels, test_logits, test_labels, val_prop, random_state)

    # Init trainer
    trainer = L.Trainer(
        accelerator = accelerator,
        strategy = "auto",
        devices = 1,
        num_nodes = 1,
        precision = 32,
        logger = [
            TBLogger(save_dir=log_dir),
            CSVLogger(save_dir=log_dir),
        ],
        max_epochs = max_epochs,
        check_val_every_n_epoch = 1,
        enable_checkpointing = False,
        enable_progress_bar = True,
        enable_model_summary = True,
        deterministic = True,
        profiler = None,
        default_root_dir = output_dir,
    )

    # Init base model
    if method == "dp_calibration":
        alpha, beta = "scalar", True
    elif method == "temp_scaling":
        alpha, beta = "scalar", False
    elif method == "bias_only":
        alpha, beta = "none", True
    else:
        raise ValueError(f"Invalid method: {method}")
    with trainer.init_module():
        model = AffineCalibration(
            num_classes = data["train"].dataset[0]["logits"].shape[0],
            alpha = alpha,
            beta = beta,
            max_ls = max_ls,
            learning_rate = learning_rate,
        )
            
    # -------------------
    # Fit the model
    # -------------------
    last_checkpoint_path = output_dir / "last.ckpt" if (output_dir / "last.ckpt").exists() else None
    trainer.fit(model, train_dataloaders=data["train"], val_dataloaders=data["val"], ckpt_path=last_checkpoint_path)
    if trainer.state.status == TrainerStatus.INTERRUPTED:
        print("Training interrupted.")
        return
    torch.save(model.state_dict(), checkpoint_dir / "model.pth")

    # -------------------
    # Evaluate the model
    # -------------------
    best_ckpt_path = output_dir / "best.ckpt"
    if not best_ckpt_path.exists():
        best_ckpt_path = output_dir / "last.ckpt"
    checkpoint = lazy_load(best_ckpt_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
        
    for split, dataloader in data.items():
        if dataloader is None:
            continue
        trainer.predict(model, dataloaders=dataloader)
        if trainer.state.status == TrainerStatus.INTERRUPTED:
            print("Prediction interrupted.")
            return
        for k, v in model.predict_outputs.items():
            if k == "idx":
                continue
            if k == "label":
                v = v.numpy().astype(int)
            else:
                v = v.numpy()
            d = pd.DataFrame(v, index=model.predict_outputs["idx"].numpy().astype(int))
            d.to_csv(output_dir / f"{split}_{k}.csv", index=True, header=False)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)