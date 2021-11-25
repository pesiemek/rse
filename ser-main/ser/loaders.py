from ser.constants import DATA_DIR
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Dict
from ser.params import Params   
import json

def load_training(batch_size: int, transform):
    return DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

def load_validation(batch_size: int, transform):
    return DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

def load_test(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    return DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)



def load_params(path):
    file = open(path)
    return _undict(json.load(file))


def _undict(params_dict: Dict): 
    params = Params(name=params_dict["name"], 
    epochs=params_dict["epochs"],
    batch_size=params_dict["batch_size"],
    learning_rate=params_dict["learning_rate"])
    return params