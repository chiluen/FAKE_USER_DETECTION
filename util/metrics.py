import torch

def accuracy(preds, labels):
    return (torch.sum(torch.argmax(preds, axis=1) == labels) / labels.shape[0]).item()