# dataloaders
from torch.utils.data import DataLoader
from torchvision import datasets

def load_data(DATA_DIR, ts, p):

    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=p.batch_size,
        shuffle=False,
        num_workers=1,
    )

    return training_dataloader, validation_dataloader