from pathlib import Path
from typing import Any, Literal
from collections import defaultdict

import lightning as L
import torch
from torch import nn
from lightning.pytorch.trainer.states import RunningStage

from datasets import Dataset

class AffineCalibrator(nn.Module):
    """
    Affine calibration block. It is a linear block that performs an affine transformation
    of the input feature vector ino order to output the calibrated logits.

    Parameters
    ----------
    num_features : int
        Number of input features of the calibrator.
    num_classes : int
        Number of output classes of the calibrator.
    alpha : {"vector", "scalar", "matrix", "none"}, optional
        Type of affine transformation, by default "vector"
    beta : bool, optional
        Whether to use a beta term, by default True
    """    
    def __init__(
        self, 
        num_classes: int, 
        alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        beta: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.additional_arguments = {
            "alpha": alpha,
            "beta": beta,
        }

        # Set the alpha parameter
        if alpha == "matrix":
            self.alpha = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        elif alpha == "vector":
            self.alpha = nn.Parameter(torch.ones(num_classes), requires_grad=True)
        elif alpha == "scalar":
            self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=True)
        elif alpha == "none":
            self.alpha = nn.Parameter(torch.tensor(1.), requires_grad=False)
        else:
            raise ValueError(f"Invalid alpha: {alpha}")
        
        # Set the beta parameter
        self.beta = nn.Parameter(torch.zeros(num_classes), requires_grad=beta)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.alpha.shape == (self.num_classes, self.num_classes):
            output = logits @ self.alpha.T
        elif self.alpha.shape in [(self.num_classes,), ()]:
            output = logits * self.alpha
        output = output + self.beta

        return output


class AffineCalibration(L.LightningModule):

    def __init__(
        self,
        num_classes: int,
        alpha: Literal["matrix", "vector", "scalar", "none"] = "matrix",
        beta: bool = True,
        max_ls: int = 40,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.max_ls = max_ls
        self.learning_rate = learning_rate

        self.calibrator = AffineCalibrator(self.num_classes, self.alpha, self.beta)
        
        self.super_global_step = 0
        self.best_val_loss = float("inf")
        self.last_val_loss = float("inf")
        self.patience_count = 0

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.LBFGS(trainable_params, lr=self.learning_rate, max_iter=self.max_ls)
    
    def forward(self, logits) -> torch.Tensor:
        return self.calibrator(logits)
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.patience_count >= 10 and self.current_epoch > 0:
            self.trainer.should_stop = True
            return -1
    
    def training_step(self, batch, batch_idx):

        optimizer = self.optimizers()
        logits, labels = batch["logits"].float(), batch["label"].long()

        def closure():
            cal_logits = self(logits)
            loss = torch.nn.functional.cross_entropy(cal_logits, labels)
            optimizer.zero_grad()
            self.manual_backward(loss)
            for logger in self.loggers:
                logger.log_metrics({
                    "train/cross_entropy": loss.item(),
                }, step=self.super_global_step)
            self.super_global_step += 1
            return loss
        
        loss = optimizer.step(closure)
        return {"loss": loss}
        
    def on_train_epoch_end(self):
        self.trainer.save_checkpoint(
            Path(self.trainer.default_root_dir) / f"last.ckpt"
        )

    def validation_step(self, batch, batch_idx):
        logits, labels = batch["logits"].float(), batch["label"].long()
        cal_logits = self(logits)
        loss = torch.nn.functional.cross_entropy(cal_logits, labels)
        
        if self.trainer.state.stage != RunningStage.SANITY_CHECKING:
            ce = loss.item()
            self.last_val_loss = ce
            for logger in self.loggers:
                logger.log_metrics({
                    "val/cross_entropy": ce,
                }, step=self.super_global_step)
            if self.best_val_loss - ce > 1e-5:
                self.patience_count = 0
                self.best_val_loss = ce
                self.trainer.save_checkpoint(
                    Path(self.trainer.default_root_dir) / "best.ckpt"
                )
            else:
                self.patience_count += 1
        return {"loss": loss}

    def on_save_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        checkpoint["super_global_step"] = self.super_global_step
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["last_val_loss"] = self.last_val_loss
        checkpoint["patience_count"] = self.patience_count

    def on_load_checkpoint(self, checkpoint: torch.Dict[str, Any]) -> None:
        self.super_global_step = checkpoint["super_global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.last_val_loss = checkpoint["last_val_loss"]
        self.patience_count = checkpoint["patience_count"]

    def on_predict_start(self) -> None:
        self.eval()

    def on_predict_epoch_start(self) -> None:
        self.predict_outputs = defaultdict(list)
    
    def predict_step(self, batch, batch_idx):
        logits, labels = batch["logits"].float(), batch["label"].long()
        cal_logits = self(logits)
        return {"idx": batch["idx"], "logits": cal_logits, "label": labels}

    def on_predict_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        for k, v in outputs.items():
            self.predict_outputs[k].append(v.cpu())

    def on_predict_end(self) -> None:
        predict_outputs = {}
        for k, v in self.predict_outputs.items():
            predict_outputs[k] = torch.cat(v, dim=0)
        self.predict_outputs = predict_outputs
    