import torch.nn as nn


def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def unfreeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train(True)
