import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
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
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1e-2, max_iter=40)

        priors = torch.bincount(labels, minlength=logprobs.shape[1]).float() / len(labels)
        priors_ce = -torch.log(priors[labels]).mean().item()

        last_nce = float("inf")
        while True:

            def closure():
                optimizer.zero_grad()
                cal_logits = self(logprobs)
                loss = F.cross_entropy(cal_logits, labels)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)

            nce = loss.item() / priors_ce
            if abs(last_nce - nce) < 1e-5:
                break
            last_nce = nce

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
