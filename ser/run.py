import json
import torch

def track_run(name, epochs, batch_size, learning_rate, model):

    torch.save(model.state_dict(), f"experiments/{name}.pth")

    hyperparameters = {
        "name": name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    with open(f"experiments/{name}.json", 'w') as f:
        json.dump(hyperparameters, f)
