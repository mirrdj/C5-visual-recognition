import torch
from tqdm import tqdm
from pathlib import Path

def validation(model, dataloader, loss_fn, device, params, contains_text):
    model.eval()
    
    val_loss, val_acc = 0.0, 0.0

    if params["save_predictions"]:
        preds_file = Path("./predictions").joinpath(f'{params["model_name"]}_validation').with_suffix(".csv")
        with open(preds_file, "w") as f:
            f.write("VideoName,ground_truth,prediction\n")

    with torch.no_grad():
        for data in tqdm(dataloader):
            names = data[0]
            inputs = data[1:-1]
            labels = data[-1]

            end = -1 if contains_text else None
            inputs[:end] = [input.to(device) for input in inputs[:end]]

            labels = labels.to(device)

            outputs = model(*inputs)
            
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * outputs.size(0)

            _, predicted = torch.max(outputs, 1)
            val_acc += (predicted == labels).sum().item()

            if params["save_predictions"]:
                with open(preds_file, "a") as f:
                    for i, name in enumerate(names):
                        f.write(f"{name},{dataloader.dataset.CLASSES[labels[i]]},{dataloader.dataset.CLASSES[predicted[i]]}\n")

    val_loss /= len(dataloader.dataset)
    val_acc /= len(dataloader.dataset)

    return val_loss, val_acc