from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import alexnet

from ser.model import Net
from ser.train import train_loop
from ser.data import load_data
from ser.transforms import transform_data
from ser.run import track_run
from ser.hyperparameters import Params


import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        ..., "-b", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        ..., "-r", "--learning_rate", help="Learning Rate"
    )
):
    print(f"Running experiment {name}")
    print(f"{epochs} epochs")
    print(f"Batch size {batch_size}")
    print(f"Learning rate {learning_rate}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    p = Params(name, epochs, batch_size, learning_rate)

    # load model
    model = Net().to(device)
    

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=p.learning_rate)

    # torch transforms
    ts = transform_data()

    # dataloaders
    training_dataloader, validation_dataloader = load_data(DATA_DIR, ts, p)

    # train
    save_val_acc = train_loop(p, training_dataloader, validation_dataloader, device, model, optimizer)

    track_run(p, model, save_val_acc)


@main.command()
def infer():
    print("This is where the inference code will go")
