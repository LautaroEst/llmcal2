
import numpy as np
from scipy.special import softmax, log_softmax
from .calibration import train_cal_on_test, calibrate_xval

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

def compute_cal_loss(logits, labels, mode="trainontest", metric="nce"):
    if mode == "trainontest":
        cal_logprobs = train_cal_on_test(logits, labels)
    elif mode == "xval":
        cal_logprobs = calibrate_xval(logits, labels, seed=1234, condition_ids=None, stratified=True, nfolds=5) 
    else:
        raise ValueError(f"Unknown mode: {mode}")
    nce = compute_metric(logits, labels, metric)
    cal_nce = compute_metric(cal_logprobs, labels, metric)
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
    elif "calloss" in metric:
        _, metric, mode = metric.split("_")
        return compute_cal_loss(logits, labels, mode, metric)
    elif metric == "ece":
        return compute_ece(logits, labels)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    

def compute_psr_with_mincal(logits, labels, psr, mode):
    if mode == "trainontest":
        cal_logprobs = train_cal_on_test(logits, labels)
    elif mode == "xval":
        cal_logprobs = calibrate_xval(logits, labels, seed=1234, condition_ids=None, stratified=True, nfolds=5) 
    elif mode == "none":
        cal_logprobs = logits
    else:
        raise ValueError(f"Unknown mode: {mode}")
    loss = compute_metric(logits, labels, psr)
    cal_loss = compute_metric(cal_logprobs, labels, psr)

    return loss, cal_loss