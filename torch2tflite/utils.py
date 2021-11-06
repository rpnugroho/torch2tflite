import os
import torch
from torchvision import models


def get_filename(path: str):
    """Get filename from path

    Args:
        path (string): Model path

    Returns:
        String: Filename
    """
    return os.path.basename(path).split(".")[0]


def load_model_from_file(path: str):
    """Load model from file path

    Args:
        path (str): Model path

    Returns:
        Pytorch model
    """
    return torch.load(path)


def load_torchvision_model(model_name: str, pretrained: bool = True):
    """Load model directly from torchvision

    Args:
        model_name (str): Model name
        pretrained (bool, optional): Pretrainde using imagenet. Defaults to True.

    Returns:
        Pytorch model
    """
    return getattr(models, model_name)(pretrained=pretrained)
