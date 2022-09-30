from torchvision import datasets, transforms

def transform_data():
    ts = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    return ts