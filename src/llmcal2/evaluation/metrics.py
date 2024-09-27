
import numpy as np
from scipy.special import softmax, log_softmax
from .calibration import train_cal_on_test

def compute_ner(logits, labels):
    er = (logits.argmax(axis=1) != labels).mean()
    winner = np.bincount(labels, minlength=logits.shape[1]).argmax()
    norm = (labels != winner).mean()
    return er / norm

def compute_nce(logits, labels):
    ce = -log_softmax(logits, axis=1)[np.arange(len(labels)), labels].mean()
    priors = np.bincount(labels, minlength=logits.shape[1]) / len(labels)
    norm = -np.log(priors[labels]).mean()
    return ce / norm

def compute_nbrier(logits, labels):
    one_hot = np.zeros(logits.shape)
    one_hot[np.arange(len(labels)), labels] = 1
    brier = ((one_hot - softmax(logits, axis=1))**2).mean()
    priors = np.bincount(labels, minlength=logits.shape[1]) / len(labels)
    norm = ((one_hot - priors)**2).mean()
    return brier / norm

def compute_cal_loss_nce(logits, labels):
    cal_logprobs = train_cal_on_test(logits, labels)
    nce = compute_nce(logits, labels)
    cal_nce = compute_nce(cal_logprobs, labels)
    return (nce - cal_nce) / nce

def compute_ece(logits, labels):
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = softmax(logits, axis=1)
    confidences = softmaxes.max(axis=1)
    predictions = softmaxes.argmax(axis=1)
    accuracies = predictions == labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def compute_metric(logits, labels, metric):
    if metric == "ner":
        return compute_ner(logits, labels)
    elif metric == "nce":
        return compute_nce(logits, labels)
    elif metric == "nbrier":
        return compute_nbrier(logits, labels)
    elif metric == "cal_loss_nce":
        return compute_cal_loss_nce(logits, labels)
    elif metric == "ece":
        return compute_ece(logits, labels)
    else:
        raise ValueError(f"Unknown metric: {metric}")