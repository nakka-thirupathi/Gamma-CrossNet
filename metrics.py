import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)

def classification_metrics(y_true, prob):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    try: auc = roc_auc_score(y_true, prob[:,1])
    except ValueError: auc = float("nan")
    return {
      "accuracy": accuracy_score(y_true, pred),
      "precision": precision_score(y_true, pred, zero_division=0),
      "recall_sensitivity": recall_score(y_true, pred, zero_division=0),
      "specificity": tn / max(1, tn + fp),
      "f1": f1_score(y_true, pred, zero_division=0),
      "auc_roc": auc, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def segmentation_metrics(pred, target, eps=1e-7):
    p = (np.asarray(pred) >= 0.5).astype(bool)
    t = (np.asarray(target) >= 0.5).astype(bool)
    inter = np.logical_and(p,t).sum()
    union = np.logical_or(p,t).sum()
    dice = (2*inter+eps)/(p.sum()+t.sum()+eps)
    iou = (inter+eps)/(union+eps)
    return {"dice": float(dice), "iou": float(iou)}
