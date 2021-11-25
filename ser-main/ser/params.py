from dataclasses import dataclass, asdict
import json


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