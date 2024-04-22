import torch
from pathlib import Path
from tqdm import tqdm
from utils import reorder_csv

def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device, params, contains_text):

    if params["save_predictions"]:
        preds_file = Path("./predictions").joinpath(f'{params["model_name"]}').with_suffix(".csv")
        with open(preds_file, "w") as f:
            f.write("VideoName,ground_truth,prediction\n")

    # Put model in test mode
    model.eval()
    
    # Setup test loss and test accuracy values
    test_acc = 0
    
    print("Evaluating on test set...")
    # Loop through data loader data batches
    counter = 1
    with torch.no_grad():
        for data in tqdm(dataloader):
            names = data[0]
            inputs = data[1:-1]
            labels = data[-1]

            end = -1 if contains_text else None
            inputs[:end] = [input.to(device) for input in inputs[:end]]

            labels = labels.to(device)

            if(len(inputs[0])!=1):
                print("ERROR: batch size (test stage) =", len(inputs[0]))
                print("ERROR: my_test_step function require batch size = 1 to generate the correct output file")
                exit()
            
            
            # 1. Forward pass
            y_pred = model(*inputs)

            # Calculate and accumulate accuracy metric across all batches
            _, predicted = torch.max(y_pred, 1)
            test_acc += (predicted == labels).sum().item()

            if params["save_predictions"]:
                with open(preds_file, "a") as f:
                    for i, name in enumerate(names):
                        f.write(f"{name},{dataloader.dataset.CLASSES[labels[i]]},{dataloader.dataset.CLASSES[predicted[i]]}\n")

            counter +=1

    reorder_csv(preds_file)

    # Adjust metrics to get average accuracy per batch 
    test_acc /= len(dataloader.dataset)
    print("Average accuracy = ", test_acc)
    