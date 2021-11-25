from ser.art import generate_ascii_art
from ser.loaders import load_training, load_validation, load_params
from ser.model import Net
from ser.params import Params
from ser.train import run_training
from ser.transforms import normalize, transform
from ser.helpers import set_paths
import torch

from torch import optim

import typer
main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        "run", "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Option(1, "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Option(1000, "--batch", help="Size of batch for the data loader"),
    learning_rate: float = typer.Option(0.001, "--rate", help="Learning reate for the optimiser")
):

    print(f"Running experiment {name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = Net().to(device)

    # setup params and loaders
    params = Params(name, epochs, batch_size, learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    test_data = load_training(params.batch_size, transform=transform(normalize))
    val_data = load_validation(params.batch_size, transform=transform(normalize))
    
    # train
    best_model = run_training(model, epochs, optimizer, test_data, val_data, device)

    model_path, params_path = set_paths(name)
    torch.save(best_model, model_path)
    params.save(params_path)



@main.command()
def infer(
    timestamp: str = typer.Option(
        ..., "-t", "--timestamp", help="Name of experiment to infer."),

    experiment: str = typer.Option(
        "folder", "-e", "--experiment", help="Name of your experiment folder"),
    
    label: int = typer.Option(2, "-l", "--label", help="Which number you'd like to see prediction of")
    ):

    model_path, params_path = set_paths(experiment, timestamp)

    # load the parameters from the run_path so we can print them out!
    params = load_params(params_path)
    print(f"\nInference on \n model: {experiment} \n ran at: {timestamp} \n")
    params.print()
    print("")

    # select image to run inference for
    dataloader = load_training(params.batch_size, transform=transform(normalize))
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

