from dataclasses import dataclass, asdict
import json
from typing import Dict


@dataclass
class Params:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


    def print(self):
        print("Model details: ")
        for attribute in self.__dict__.keys():
                print(f" {attribute}: {self.__dict__[attribute]}")



def load_params(path):
    file = open(path)
    return _undict(json.load(file))


def _undict(params_dict: Dict): 
    params = Params(name=params_dict["name"], 
    epochs=params_dict["epochs"],
    batch_size=params_dict["batch_size"],
    learning_rate=params_dict["learning_rate"])
    return params