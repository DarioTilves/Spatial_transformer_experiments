import os
import numpy as np
import pandas as pd


def accumulate_confusion_matrix(confusion_matrix: np, targets: np, predictions: np):
    coordinates = np.stack((targets, predictions))
    array_aux = np.array([len(confusion_matrix), len(confusion_matrix)])
    flt = np.ravel_multi_index(coordinates, array_aux) 
    confusion_matrix += np.bincount(flt, minlength=array_aux.prod()).reshape(array_aux)                         
    return confusion_matrix


def calculate_metrics(confusion_matrix: np) -> dict:
    metrics = {}
    true_positives = np.diag(confusion_matrix)
    false_positives = np.sum(confusion_matrix, axis = 0) - true_positives
    false_negatives = np.sum(confusion_matrix, axis = 1) - true_positives
    true_negatives = np.sum(confusion_matrix) - true_positives - false_positives - false_negatives
    true_positives.setflags(write=1)
    true_positives[true_positives == 0] = 1
    true_negatives.setflags(write=1)
    true_negatives[true_negatives == 0] = 1
    metrics['accuracy'] = np.mean((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives))
    metrics['precision'] = np.mean(true_positives / (true_positives + false_positives))
    metrics['sensitivity'] = np.mean(true_positives / (true_positives + false_negatives))
    metrics['specifity'] = np.mean(true_negatives / (true_negatives + false_positives))
    metrics['f1score'] = np.mean(2*true_positives/ (2*true_positives + false_positives + false_negatives))
    return metrics


def save_metrics(epoch: int, save_dict: dict, save_path: str):
    save_dict['Epoch'] = epoch
    pd_metrics = pd.DataFrame(save_dict, index = [save_dict['Epoch']]) 
    epoch_col = pd_metrics.pop('Epoch')
    pd_metrics.insert(0, 'Epoch', epoch_col)
    if not os.path.isfile(save_path):
        pd_metrics.to_csv(save_path, mode='a', header=True, index=False, sep=';', decimal = ',')
    else:
        pd_metrics.to_csv(save_path, mode='a', header=False, index=False, sep=';', decimal = ',')
    return
