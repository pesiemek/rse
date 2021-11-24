from ser.constants import DATA_DIR
from torch.utils.data import DataLoader
from torchvision import datasets
   

def load_data(batch_size: int, type: str, transform):
    if type=="train":
        return DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    elif type=="validation":
        return DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    else:
        print("Data Loader type not recognised. Allowed types: train or validation")
        return
