from model import Net


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

    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
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


