
import warnings
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lbfgs import LBFGS
import torch.nn.functional as F
from typing import Literal

from ..loggers import TBLogger, CSVLogger

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
):
    torch.set_float32_matmul_precision("high")
    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)

    # Load train data
    train_logits = torch.log_softmax(torch.from_numpy(pd.read_csv(train_logits, index_col=0, header=None).values).float(), dim=1)
    train_labels = torch.from_numpy(pd.read_csv(train_labels, index_col=0, header=None).values.flatten()).long()
    train_dataset = TensorDataset(train_logits, train_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=len(train_dataset), 
        shuffle=False,
    )
    
    # Load predict data
    df_predict_logits = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_logits = torch.log_softmax(torch.from_numpy(df_predict_logits.values).float(), dim=1)
    df_predict_labels = pd.read_csv(predict_labels, index_col=0, header=None)
    predict_labels = torch.from_numpy(df_predict_labels.values.flatten()).long()

    # Train model
    model = AffineCalibrator(method=method, num_classes=train_logits.shape[1])
    optimizer = LBFGS(
        params=(param for param in model.parameters() if param.requires_grad),
        lr=learning_rate,
        max_iter=max_ls,
        tolerance_change=tolerance,
    )
    state = fit(model, optimizer, train_loader, log_dir, tolerance)

    # Predict
    cal_logits = predict(model, predict_logits)

    # Save results
    pd.DataFrame(cal_logits, index=df_predict_logits.index).to_csv(output_dir / 'logits.csv', index=True, header=False)
    df_predict_labels.to_csv(output_dir / 'labels.csv', index=True, header=False)
    torch.save(state, checkpoint_dir / 'last.ckpt')



def fit(model, optimizer, train_loader, log_dir, tolerance=1e-4):
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
        'loss': float('inf'),
        'step_count': 0,
    }
    while True:
        last_loss = state['loss']
        logits, labels = next(iter(train_loader))
        def closure():
            optimizer.zero_grad()
            cal_logits = model(logits)
            loss = F.cross_entropy(cal_logits, labels)
            er = (cal_logits.argmax(dim=1) != labels).float().mean().item()
            for logger in loggers:
                logger.log_metrics({
                    "train/NCE": loss.item() / priors_ce,
                    "train/CE": loss.item(),
                    "train/NER": er / priors_er,
                    "train/ER": er,
                }, step=state['step_count'])
            loss.backward()
            state['step_count'] += 1
            return loss
        loss = optimizer.step(closure)
        state['model'] = model.state_dict()
        state['loss'] = loss.item()
        if abs(state['loss'] - last_loss) < tolerance:
            break
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