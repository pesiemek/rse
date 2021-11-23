from pathlib import Path
from ser.data import dataloader
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    device: torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Argument(..., "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Argument(..., "--batch", help="Size of batch for the data loader"),
    learning_rate: int = typer.Argument(..., "--rate", help="Learning reate for the optimiser"),

    
):
    print(f"Running experiment {name}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # epochs = 2
    # batch_size = 1000
    # learning_rate = 0.01

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

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(test_data):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
            val_loss = 0
            correct = 0
        with torch.no_grad():
            for images, labels in validation_dataloader:
                  images, labels = images.to(device), labels.to(device)
                  model.eval()
                  output = model(images)
                  val_loss += F.nll_loss(output, labels, reduction="sum").item()
                  pred = output.argmax(dim=1, keepdim=True)
                  correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss /= len(validation_dataloader.dataset)
            val_acc = correct / len(validation_dataloader.dataset)

            print(
                 f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@main.command()
def infer():
    print("This is where the inference code will go")
