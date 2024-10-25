import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold, KFold

class DPCalibrator(nn.Module):
    
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x):
        return self.alpha * x + self.beta
    
    def calibrate(self, logprobs):
        self.eval()
        with torch.no_grad():
            cal_logprobs = torch.log_softmax(self(logprobs), dim=1)
        return cal_logprobs
    
    def fit(self, logprobs, labels):
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2, max_iter=100)

        def closure():
            optimizer.zero_grad()
            cal_logprobs = self(logprobs)
            loss = criterion(cal_logprobs, labels)
            loss.backward()
            return loss
        
        last_loss = float("inf")
        loss = optimizer.step(closure)
        while abs(last_loss - loss) < 1e-6:
            loss = optimizer.step(closure)

        return self


def train_cal_on_test(logits, labels):
    calibrator = DPCalibrator(n_classes=logits.shape[1])
    logprobs = torch.log_softmax(torch.from_numpy(logits).float(), dim=1)
    labels = torch.from_numpy(labels).long()
    calibrator.fit(logprobs, labels)
    calibrated_logprobs = calibrator.calibrate(logprobs).numpy()
    return calibrated_logprobs


def calibrate_xval(logits, targets, seed=0, condition_ids=None, stratified=True, nfolds=5):
    logprobs = torch.log_softmax(torch.from_numpy(logits).float(), dim=1)
    targets = torch.from_numpy(targets).long()
    logprobscal = torch.zeros(logprobs.size())
    
    if stratified:
        if condition_ids is not None:
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        else:
            skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    else:
        if condition_ids is not None:
            skf = GroupKFold(n_splits=nfolds)
        else:
            skf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    for trni, tsti in skf.split(logprobs, targets, condition_ids):
        model = DPCalibrator(n_classes=logprobs.shape[1])
        model.fit(logprobs[trni], targets[trni])
        with torch.no_grad():
            logprobscal[tsti] = torch.log_softmax(model.forward(logprobs[tsti]), dim=1)

    return logprobscal
