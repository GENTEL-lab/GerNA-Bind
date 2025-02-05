"""
@author: Yunpeng Xia
"""

from sklearn.metrics import roc_curve,auc,roc_auc_score,classification_report,precision_recall_curve,f1_score,average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import math

def auc_curve(prob,y):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("DRIM_AUROC.png")

def auprc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("DRIM_AUPRC.png")

def get_train_metrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]

    TP = FP = 0
    TN = np.sum(y_true == 0) 
    FN = np.sum(y_true == 1) 
    mcc = 0
    mcc_threshold = y_pred[0] + 1 
    confuse_matrix = (TP, FP, TN, FN) 
    max_mcc = -1 

    for index, score in enumerate(y_pred):
        if y_true[index] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        numerator = (TP * TN - FP * FN)
        denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator

        if mcc > max_mcc:
            max_mcc = mcc
            confuse_matrix = (TP, FP, TN, FN)
            mcc_threshold = score  # 
    TP, FP, TN, FN = confuse_matrix  # 
    Pre = 0 if (TP + FP) == 0 else (TP / (TP + FP))
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)
    F1_score = 0 if (Pre + Sen) == 0 else ( 2 * Pre * Sen / (Pre + Sen) )
    AUPRC = average_precision_score(y_true, y_pred)
    return mcc_threshold, TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, max_mcc, AUC, AUPRC

def get_valid_metrics(y_pred, y_true, threshold):
    # print(threshold)
    # print(y_pred)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= threshold:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] < threshold:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] >= threshold:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] < threshold:
            TN += 1
    Pre = 0 if (TP + FP) == 0 else (TP / (TP + FP))
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)

    F1_score = 0 if (Pre + Sen) == 0 else ( 2 * Pre * Sen / (Pre + Sen) )
    AUPRC = average_precision_score(y_true, y_pred)
    
    numerator = (TP * TN - FP * FN)
    denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator
    return TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC

def get_test_metrics(y_pred, y_true, threshold):
    # print(threshold)
    # print(y_pred)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    TP = TN = FP = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] >= threshold:
            TP += 1
        elif y_true[i] == 1 and y_pred[i] < threshold:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] >= threshold:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] < threshold:
            TN += 1
    Pre = 0 if (TP + FP) == 0 else (TP / (TP + FP))
    Sen = 0 if (TP + FN) == 0 else (TP / (TP + FN))
    Spe = 0 if (TN + FP) == 0 else (TN / (TN + FP))
    Acc = 0 if (TP + FP + TN + FN) == 0 else ((TP + TN) / (TP + FP + TN + FN))
    AUC = roc_auc_score(y_true, y_pred)

    F1_score = 0 if (Pre + Sen) == 0 else ( 2 * Pre * Sen / (Pre + Sen) )
    AUPRC = average_precision_score(y_true, y_pred)
    
    #AUC = auc_cal(y_pred,y_true)
    auc_curve(y_pred,y_true)
    auprc_curve(y_true, y_pred)
    numerator = (TP * TN - FP * FN)
    denominator = (math.sqrt(TP + FP) * math.sqrt(TN + FN) * math.sqrt(TP + FN) * math.sqrt(TN + FP))
    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator
    return TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC