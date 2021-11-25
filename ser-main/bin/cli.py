from typing import List
import torch
import typer
from ser.helpers import set_paths
from ser.infer import run_inference
from ser.loaders import load_params, load_training, load_validation
from ser.model import Net
from ser.params import Params
from ser.train import run_training
from ser.transforms import normalize, flip, transform
from torch import optim

main = typer.Typer()


@main.command()
def train(
    name: str = typer.Option(
        "run", "-n", "--name", help="Name of experiment to save under."

    ),
    epochs: int = typer.Option(1, "--epochs", help="Number of epochs to run"),
    batch_size: int = typer.Option(1000, "--batch", help="Size of batch for the data loader"),
    learning_rate: float = typer.Option(0.001, "--rate", help="Learning reate for the optimiser"),
    transforms: List[str] = typer.Option(["normalize"], "-t", "--transforms", help="A list of transforms to perform on the data. Available options are 'normalize' and 'flip'")
):

    print(f"Running experiment {name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = Net().to(device)

    # setup params and loaders
    params = Params(name, epochs, batch_size, learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    transforms_composition = transform(*transforms)
    
    train_data = load_training(params.batch_size, transform=transforms_composition)
    val_data = load_validation(params.batch_size, transform=transforms_composition)
    
    # train
    best_model = run_training(model, epochs, optimizer, train_data, val_data, device)

    model_path, params_path = set_paths(name)
    torch.save(best_model, model_path)
    params.save(params_path)



@main.command()
def infer(
    timestamp: str = typer.Option(
        ..., "-t", "--timestamp", help="Name of experiment to infer."),

    experiment: str = typer.Option(
        "bam", "-e", "--experiment", help="Name of your experiment folder"),
    
    label: int = typer.Option(2, "-l", "--label", help="Which number you'd like to see prediction of"),
    flip_image: bool = typer.Option(False, "--flip")
    ):

    model_path, params_path = set_paths(experiment, timestamp)

    # load the parameters from the run_path so we can print them out!
    params = load_params(params_path)
    print(f"\nInference on \n model: {experiment} \n ran at: {timestamp} \n")
    params.print()
    print("")

    model = torch.load(model_path)
    
    if flip_image:
        ts = [normalize, flip]
    else:
        ts = [normalize]

    run_inference(model, label, ts)

