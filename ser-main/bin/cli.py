from pathlib import Path
from ser.data import dataloader
from ser.model import Net
from ser.train import train
from ser.validate import validate
import torch

from torch import optim
import torch.nn.functional as F
from torchvision import  transforms

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def training(
    device: torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Argument(..., "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Argument(..., "--batch", help="Size of batch for the data loader"),
    learning_rate: int = typer.Argument(..., "--rate", help="Learning reate for the optimiser")
):

    print(f"Running experiment {name}")

    # save the parameters!

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

    # train
    for epoch in range(epochs):
        model, loss = train(model, optimizer, test_data, device)

        print(f"Train Epoch: {epoch} "
                f"| Loss: {loss.item():.4f}")

        # validate
        val_loss, val_acc = validate(model, val_data, device) 

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")



@main.command()
def infer():
    print("This is where the inference code will go")
