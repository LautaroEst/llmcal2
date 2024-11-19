
import os
import warnings
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lbfgs import LBFGS
import torch.nn.functional as F
from typing import Literal

from ..loggers import TBLogger, CSVLogger
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold

warnings.filterwarnings("ignore", category=UserWarning, message=".*Experiment logs directory outputs*")



class AffineCalibrator(torch.nn.Module):

    def __init__(self, method: str, num_classes: int):
        super().__init__()
        self.method = method
        self.num_classes = num_classes
        self._init_params(method)

    def _init_params(self, method):
        if method == "dp_calibration":
            self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)
        elif method == "temp_scaling":
            self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.zeros(self.num_classes), requires_grad=False)
        elif method == "bias_only":
            self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.beta = torch.nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)
        else:
            raise ValueError(f"Invalid method: {method}")
        
    def forward(self, logits):
        return logits * self.alpha + self.beta



def main(
    output_dir: str = 'output',
    log_dir: str = 'output/logs',
    checkpoint_dir: str = 'output/checkpoints',
    train_logits: str = 'logits.csv',
    train_labels: str = 'labels.csv',
    predict_logits: str = 'logits.csv',
    predict_labels: str = 'labels.csv',
    method: Literal["dp_calibration", "temp_scaling", "bias_only"] = "dp_calibration",
    learning_rate: float = 1e-3,
    tolerance: float = 1e-4,
    max_ls: int = 100,
    seed: int = 0,
):
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load train data
    train_logits = torch.log_softmax(torch.from_numpy(pd.read_csv(train_logits, index_col=0, header=None).values).float(), dim=1)
    train_labels = torch.from_numpy(pd.read_csv(train_labels, index_col=0, header=None).values.flatten()).long()
    
    # Load predict data
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_logits = torch.log_softmax(torch.from_numpy(df_predict_logits.values).float(), dim=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = torch.from_numpy(df_predict_labels.values.flatten()).long()

    state = fit(method, train_logits, train_labels, log_dir, tolerance, train_logits.shape[1], learning_rate, max_ls, seed)

    # Predict
    model = AffineCalibrator(method=method, num_classes=train_logits.shape[1])
    model.load_state_dict(state['model'])
    cal_logits = predict(model, predict_logits)

    # Save results
    pd.DataFrame(cal_logits, index=df_predict_logits.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)
    torch.save(state, checkpoint_dir / 'last.ckpt')


def fit(method, logits, labels, log_dir, tolerance, num_classes, learning_rate, max_ls, seed):
    
    # Create folds
    steps = []
    rs = torch.Generator().manual_seed(seed)
    for i in range(5):
        ids = torch.randperm(logits.shape[0], generator=rs)
        trni = ids[:int(0.7*len(ids))]
        tsti = ids[int(0.7*len(ids)):]

        # Train model
        model = AffineCalibrator(method=method, num_classes=num_classes)
        optimizer = LBFGS(
            params=(param for param in model.parameters() if param.requires_grad),
            lr=learning_rate,
            max_iter=max_ls,
            tolerance_change=tolerance,
        )
        train_dataset = TensorDataset(logits[trni], labels[trni])
        train_loader = DataLoader(
            train_dataset, 
            batch_size=len(train_dataset), 
            shuffle=False,
        )
        val_dataset = TensorDataset(logits[tsti], labels[tsti])
        val_loader = DataLoader(
            val_dataset, 
            batch_size=len(val_dataset), 
            shuffle=False,
        )
        state = _fit_to_fold(model, optimizer, train_loader, val_loader, os.path.join(log_dir,f"fold_{i}"), float('inf'), tolerance, patience=10)
        steps.append(state['step_count'])
    
    print(f"Fitting final model with {max(steps)} steps. All steps: {steps}")
    model = AffineCalibrator(method=method, num_classes=num_classes)
    optimizer = LBFGS(
        params=(param for param in model.parameters() if param.requires_grad),
        lr=learning_rate,
        max_iter=max_ls,
        tolerance_change=tolerance,
    )
    train_dataset = TensorDataset(logits[trni], labels[trni])
    train_loader = DataLoader(
        train_dataset, 
        batch_size=len(train_dataset), 
        shuffle=False,
    )
    state = _fit_to_fold(model, optimizer, train_loader, None, os.path.join(log_dir,'final'), max(steps), tolerance, patience=None)
    return state

@torch.no_grad()
def validate(model, val_loader):
    logits, labels = next(iter(val_loader))
    cal_logits = model(logits)
    loss = F.cross_entropy(cal_logits, labels)
    er = (cal_logits.argmax(dim=1) != labels).float().mean().item()
    return loss.item(), er

def _fit_to_fold(model, optimizer, train_loader, val_loader, log_dir, max_step_count, tolerance=1e-4, patience=10):
    if val_loader is None:
        val_loader = train_loader

    model.train()
    loggers = [
        TBLogger(log_dir),
        CSVLogger(log_dir),
    ]
    logits, labels = next(iter(train_loader))
    priors = torch.bincount(labels, minlength=logits.shape[1]).float() / len(labels)
    priors_ce = -torch.log(priors[labels]).mean().item()
    priors_er = (priors.argmax() != labels).float().mean().item()

    state = {
        'model': model.state_dict(),
        'best_val_loss': float('inf'),
        'step_count': 0,
        'best_step_count': 0,
        'patience': 0,
    }
    while state['step_count'] < max_step_count:

        logits, labels = next(iter(train_loader))
        def closure():
            optimizer.zero_grad()
            cal_logits = model(logits)
            loss = F.cross_entropy(cal_logits, labels)
            er = (cal_logits.argmax(dim=1) != labels).float().mean().item()
            for logger in loggers:
                logger.log_metrics({
                    "train/NCE": loss.item() / priors_ce,
                    "train/NER": er / priors_er,
                }, step=state['step_count'])
            loss.backward()
            state['step_count'] += 1
            return loss
        
        optimizer.step(closure)
        
        val_loss, val_er = validate(model, val_loader)
        norm_val_loss = val_loss / priors_ce
        for logger in loggers:
            logger.log_metrics({
                "val/NCE": norm_val_loss,
                "val/NER": val_er / priors_er,
            }, step=state['step_count'])
        
        if abs(state['best_val_loss'] - norm_val_loss) <= tolerance and patience is not None:
            if state['patience'] >= patience:
                break
            state['patience'] += 1
        else:
            state['model'] = model.state_dict()
            state['best_val_loss'] = norm_val_loss
            state['best_step_count'] = state['step_count']
    return state

@torch.no_grad()
def predict(model, logits):
    model.eval()
    cal_logits = model(logits)
    cal_logits = torch.log_softmax(cal_logits, dim=1).numpy()
    return cal_logits
    

if __name__ == '__main__':
    from fire import Fire
    Fire(main)