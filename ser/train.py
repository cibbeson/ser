from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def train_loop(epochs, training_dataloader, validation_dataloader, device, model, optimizer):

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
                val_length = len(validation_dataloader.dataset)
                for images, labels in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    model.eval()
                    output = model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item() 
                val_loss /= val_length
                val_acc = correct / val_length

                print(
                    f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )
    
    