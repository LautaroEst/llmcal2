import json
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
from scipy.special import log_softmax, softmax, logsumexp
from scipy.optimize import minimize


class IterativeCalibration:

    def __init__(self, num_classes, tolerance=1e-5):
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.alpha = 1
        self.beta = np.zeros(num_classes)
        self.history = []

    def fit(self, alpha_train_logprobs, alpha_train_labels, beta_train_logprobs, beta_train_labels):
        last_loss = float('inf')
        while True:
            new_alpha = self._next_alpha(alpha_train_logprobs, alpha_train_labels, self.alpha, self.beta)
            new_beta = self._next_beta(beta_train_logprobs, beta_train_labels, new_alpha, self.beta)
            loss = np.abs(new_alpha - self.alpha) + np.linalg.norm(new_beta - self.beta)
            self.alpha = new_alpha
            self.beta = new_beta
            self.history.append({'alpha': float(self.alpha.item()), 'beta': self.beta.tolist(), 'loss': float(loss.item())})
            if np.abs(last_loss - loss) < self.tolerance:
                break
            last_loss = loss
            print()
    
    def _next_alpha(self, logprobs, labels, alpha, beta):
        
        def compute_alpha_loss(a):
            ce = -np.mean(logprobs[np.arange(len(logprobs)), labels])
            calprobs = softmax(a * logprobs + beta, axis=1)
            soft_ce = -np.mean(np.sum(calprobs * logprobs, axis=1))
            return np.abs(soft_ce - ce)

        res = minimize(compute_alpha_loss, alpha, method='L-BFGS-B', tol=self.tolerance)
        return res.x

    # def _next_beta(self, logprobs, labels, alpha, beta):
    #     logpriors = np.log(np.bincount(labels, minlength=self.num_classes) / len(labels))
    #     logmean = np.log(np.mean(np.exp(alpha * logprobs) / np.sum(np.exp(alpha * logprobs + beta), axis=1, keepdims=True), axis=0))
    #     return logpriors - logmean
    def _next_beta(self, logprobs, labels, alpha, beta):

        def compute_beta_loss(b):
            priors = np.bincount(labels, minlength=self.num_classes) / len(labels)
            mean_cal_posteriors = np.mean(softmax(alpha * logprobs + b, axis=1), axis=0)
            return np.linalg.norm(priors - mean_cal_posteriors)
        
        res = minimize(compute_beta_loss, beta, method='L-BFGS-B', tol=self.tolerance)
        return res.x

    
    def calibrate(self, logprobs):
        return log_softmax(self.alpha * logprobs + self.beta, axis=1)
    

def main(
    checkpoint_dir: str,
    output_dir: str = 'output',
    train_alpha_logits: str = 'logits.csv',
    train_alpha_labels: str = 'labels.csv',
    train_beta_logits: str = 'logits.csv',
    train_beta_labels: str = 'labels.csv',
    predict_logits: str = 'logits.csv',
    predict_labels: str = 'labels.csv',
    tolerance: float = 1e-4,
):
    train_alpha_logits = pd.read_csv(train_alpha_logits, index_col=0, header=None).values.astype(float)
    train_alpha_labels = pd.read_csv(train_alpha_labels, index_col=0, header=None).values.astype(int).flatten()
    train_alpha_logprobs = log_softmax(train_alpha_logits, axis=1)
    train_beta_logits = pd.read_csv(train_beta_logits, index_col=0, header=None).values.astype(float)
    train_beta_labels = pd.read_csv(train_beta_labels, index_col=0, header=None).values.astype(int).flatten()
    train_beta_logprobs = log_softmax(train_beta_logits, axis=1)
    predict_logits_df = pd.read_csv(predict_logits, index_col=0, header=None)
    predict_logits = predict_logits_df.values.astype(float)
    predict_logprobs = log_softmax(predict_logits, axis=1)

    num_classes = train_alpha_logits.shape[1]
    calibrator = IterativeCalibration(num_classes, tolerance=tolerance)
    calibrator.fit(train_alpha_logprobs, train_alpha_labels, train_beta_logprobs, train_beta_labels)
    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / 'history.jsonl', 'w') as f:
        for entry in calibrator.history:
            f.write(json.dumps(entry) + '\n')
    calibrated_logprobs = calibrator.calibrate(predict_logprobs)
    
    output_dir = Path(output_dir)
    pd.DataFrame(calibrated_logprobs, index=predict_logits_df.index).to_csv(output_dir / 'logits.csv', header=None)
    pd.read_csv(predict_labels, index_col=0, header=None).to_csv(output_dir / 'labels.csv', header=None)


    


if __name__ == "__main__":
    from fire import Fire
    Fire(main)