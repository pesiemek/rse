from dataclasses import dataclass, asdict
import json
from typing import Dict


@dataclass
class Params:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float


def save_params(run_path, params):
    with open(run_path, "w") as f:
        json.dump(asdict(params), f, indent=2)


def load_params(path):
    file = open(path)
    return _undict(json.load(file))


def print_params(params: Params):
    print("Model details: ")
    for attribute in dir(params):
        if not attribute.startswith('__'):
            print("{attribute}: {value}".format(attribute=attribute, value=getattr(params, attribute)))


def _undict(params_dict: Dict): 
    params = Params(name=params_dict["name"], 
        epochs=params_dict["epochs"],
        batch_size=params_dict["batch_size"],
        learning_rate=params_dict["learning_rate"])
        
    return params

