from pathlib import Path
from ser.data import dataloader
from ser.model import Net
from ser.train import train
from ser.validate import validate
import torch
import json

from torch import optim
import torch.nn.functional as F
from torchvision import  transforms

import typer

import time

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def training(
    name: str = typer.Option(
        "run", "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Option(2, "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Option(1000, "--batch", help="Size of batch for the data loader"),
    learning_rate: float = typer.Option(0.001, "--rate", help="Learning reate for the optimiser")
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment {name}")

    timestamp = time.now()

    DIRECTORY = PROJECT_ROOT / + name
    MODEL_NAME = timestamp
    PARAMETERS = timestamp + "_parameters.json"

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_data = dataloader(batch_size, workers=2, type="train", directory=DATA_DIR, transform=ts)
    val_data = dataloader(batch_size, workers=1, type="validation", directory=DATA_DIR, transform=ts)
    
    best_valid_loss = float('inf')

    # train
    for epoch in range(epochs):
        model, loss = train(model, optimizer, test_data, device)

        print(f"Train Epoch: {epoch} "
                f"| Loss: {loss.item():.4f}")

        # validate
        val_loss, val_acc = validate(model, val_data, device) \
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            print('saving my model, improvement in validation loss achieved')
            torch.save(model.state_dict(), DIRECTORY + "_" + MODEL_NAME)

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

    
    params = {"validation_loss": best_valid_loss, "epochs": epochs,
     "batch_size": batch_size, "learning_rate": learning_rate,
      "optimizer": "Adam", "run_name": name}

    with open(PARAMETERS, 'w') as f:
        json.dump(params, f)

@main.command()
def infer():
    print("This is where the inference code will go")
