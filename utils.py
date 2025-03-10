import os
import logging
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def init_logger(log_path):
    logger = logging.getLogger("global")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=log_path)
    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def cal_metrics(pred, actual, logger):
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, pr_thresh = metrics.precision_recall_curve(actual, pred)
    aupr_score = metrics.auc(recall, precision)
    logger.info("AUC: "+str(auc_score))
    logger.info("AUPR: "+str(aupr_score))
    f1 = 2*precision*recall/(precision+recall+1e-8)
    thresh = pr_thresh[np.argmax(f1)]
    fmax = np.max(f1)
    logger.info("Fmax: "+str(fmax))
    logger.info("thresh: "+str(thresh))