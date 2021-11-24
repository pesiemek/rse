from pathlib import Path
from ser.art import generate_ascii_art
from ser.constants import DATA_DIR, PROJECT_ROOT, TIMESTAMP_FORMAT
from ser.data import load_data
from ser.model import Net
from ser.params import Params, load_params
from ser.train import train_batch
from ser.transforms import normalize, transform
from ser.validate import validate_batch
import torch

from torch import optim

import typer
main = typer.Typer()
from datetime import datetime



@main.command()
def train(
    name: str = typer.Option(
        "run", "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Option(2, "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Option(1000, "--batch", help="Size of batch for the data loader"),
    learning_rate: float = typer.Option(0.001, "--rate", help="Learning reate for the optimiser")
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment {name}")

    params = Params(name, epochs, batch_size, learning_rate)

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    results_path = PROJECT_ROOT / "Results" / "{name}".format(name=name)
    model_path = results_path / "model_{timestamp}.pt".format(timestamp=timestamp)
    params_path = results_path / "params_{timestamp}.json".format(timestamp=timestamp)
    results_path.mkdir(parents=True, exist_ok=True)

    test_data = load_data(params.batch_size, type="train", transform=transform(normalize))
    val_data = load_data(params.batch_size, type="validation", transform=transform(normalize))
    
    best_valid_loss = float('inf')

    # train
    for epoch in range(epochs):
        model, loss = train_batch(model, optimizer, test_data, device)

        print(f"Train Epoch: {epoch} "
                f"| Loss: {loss.item():.4f}")

        # validate
        val_loss, val_acc = validate_batch(model, val_data, device) \
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            print('Saving the model, improvement in validation loss achieved')
            torch.save(model, model_path)

        print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

    params.save(params_path)



@main.command()
def infer(
    timestamp: str = typer.Option(
        ..., "-t", "--timestamp", help="Name of experiment to infer."),

    experiment: str = typer.Option(
        "bimbo", "-e", "--experiment", help="Name of your experiment folder")
    ):

    run_path = Path(PROJECT_ROOT / "Results" / "{experiment}".format(experiment=experiment))
    params_path = run_path / "params_{timestamp}.json".format(timestamp=timestamp)
    model_path = run_path / "model_{timestamp}.pt".format(timestamp=timestamp)
    label = 7

    # load the parameters from the run_path so we can print them out!
    params = load_params(params_path)
    print(f"\nInference on \n model: {experiment} \n ran at: {timestamp} \n")
    params.print()

    # select image to run inference for
    dataloader = load_data(params.batch_size, type="train", transform=transform(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(model_path)

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = torch.round(max(list(torch.exp(output)[0]))*100)
   
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"I am {certainty}% sure that this is a {pred}")

