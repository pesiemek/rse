from torch.utils.data import DataLoader
from torchvision import datasets
   

def dataloader(batch_size: int, workers: int, type: str, directory, transform): 
    if type=="train":
        return DataLoader(
        datasets.MNIST(root=directory, download=True, train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    elif type=="validation":
        return DataLoader(
        datasets.MNIST(root=directory, download=True, train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    else:
        print("Data Loader type not recognised. Allowed types: train and validation")
        return
