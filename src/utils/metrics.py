from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # return correct / len(labels)
    result = correct / len(labels)
    return result.cpu().detach().numpy().item()

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    #recall
    micro_recall = recall_score(labels, preds, average='micro')
    macro_recall = recall_score(labels, preds, average='macro')
    #precision
    micro_precision = precision_score(labels, preds, average='micro')
    macro_precision = precision_score(labels, preds, average='macro')
    return micro, macro, micro_recall, macro_recall, micro_precision, macro_precision

def f1_isr(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    print(f"preds.shape: {preds.shape}, labels.shape: {labels.shape}")
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(preds.shape[0]):
        if preds[i]==1 and labels[i]==1:
            tp += 1
        elif preds[i]==0 and labels[i]==1:
            fn += 1
        elif preds[i]==1 and labels[i]==0:
            fp += 1
        elif preds[i]==0 and labels[i]==0:
            tn += 1
        else:
            raise ValueError("the category number is incorrect")
    
    if tp+fn == 0:
        recall = tp/(tp+fn+0.0001)
    else:
        recall = tp/(tp+fn)

    if tp+fp == 0:
        precision = tp/(tp+fp+0.0001)
    else:
        precision = tp/(tp+fp)
    if recall+precision == 0:
        return 2*recall*precision /(recall + precision + 0.0001), recall, precision #because we have imbalanced data, we add a small number to avoid division by zero
    else:
        return 2*recall*precision /(recall + precision), recall, precision

def f1_my_micro(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    tp, fp, fn = 0, 0, 0
    for pred, label in zip(preds, labels):
        if pred == label:
            tp += 1
        else:
            fp += 1
            fn += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1, recall, precision 

