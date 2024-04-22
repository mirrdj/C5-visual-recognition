from pathlib import Path

import wandb
from tqdm import tqdm
import torch

from utils import get_indices


def train(model, dataloader, loss_fn, optimizer, params, device, train_iters, contains_text):
    model.train()

    train_loss, train_acc = 0.0, 0.0
    
    # Entry contains
    sample_count = 0
    train_steps = 0
    for data in tqdm(dataloader):
        train_steps += 1

        inputs = data[1:-1]
        labels = data[-1]

        end = -1 if contains_text else None
        inputs[:end] = [input.to(device) for input in inputs[:end]]

        labels = labels.to(device)
            
        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(*inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Compute training accuracy and loss
        train_loss += loss.item() * outputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_acc += (predicted == labels).sum().item()

        sample_count += outputs.size(0)
        
        wandb.log({
            'Train Loss': train_loss/sample_count,
            'Train Accuracy': train_acc/sample_count
        }, step=train_iters+train_steps)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss /= len(dataloader.dataset)
    train_acc /= len(dataloader.dataset)
    
    return train_loss, train_acc, train_steps