from pathlib import Path

import torch
from torch.utils.data import DataLoader
# from datasets.img_processing import data_augmentation
import numpy as np
import pandas as pd
import wandb

from evaluation_script.evaluate import evaluate
from utils import get_optimizer, data_augmentation
from datasets import OverSampler

# Set the random seed for Python and NumPy
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")


def run_model(params, model, train_fn, validation_fn, dataset, loss_fn, trial_number, id, test_fn, contains_text):
    model_name = f"{model.__class__.__name__}_{str(id)}_{str(trial_number)}"
    params.update({"model_name": model_name})

    model.to(device)

    if params['resampling'] == True:
        dataset_train = OverSampler(dataset(split='train'), desired_dist={i: 1 for i in range(len(dataset.CLASSES))}, label_key=lambda d,i: d.age[i])
    else:
        dataset_train = dataset(split='train')
    
    dataloader_train = DataLoader(
        dataset_train, batch_size=params['batch_size'], shuffle=True, num_workers=0)

    dataset_val = dataset(split='valid')
    dataloader_val = DataLoader(
        dataset_val, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    
    dataset_test = dataset(split='test')
    dataloader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0)

    optimizer = get_optimizer(params, model)

    epochs = params['epochs']

    # Define early stopping parameters
    patience = 30
    best = np.Inf
    current_patience = 0

    train_iters = 0
    for epoch in range(epochs):
        wandb.log(data={
            'epoch': epoch,
        }, step=train_iters)

        train_loss, train_accuracy, train_steps = train_fn(
            model, dataloader_train, loss_fn, optimizer, params, device, train_iters)
        train_iters += train_steps
        val_loss, val_accuracy = validation_fn(
            model, dataloader_val, loss_fn, device, params)

        # Early stopping
        if val_loss < best:
            best = val_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")

            torch.save(model.state_dict(), Path(
                "./weights") / model_name)  # Save weights
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss/Accuracy: {val_loss:.4f}/{val_accuracy:.4f}')

        wandb.log(data={
            'Validation Loss': val_loss,
            'Validation accuracy': val_accuracy
        }, step=train_iters)

    model.load_state_dict(torch.load(Path("./weights") / model_name))
    preds_file = Path("./predictions").joinpath(f'{params["model_name"]}').with_suffix(".csv")

    # evaluate on the test set and generate the 'predictions_test_set.csv' file (used later by the evaluation script)
    test_fn(model, dataloader_test, device, params)

    full_annotations = pd.read_csv(
        r'/ghome/group02/C5-G2/Week6/data/First_Impressions_v3_multimodal/test_set_age_labels.csv', sep=',')
    pretictions_with_gt = pd.read_csv(
        preds_file, sep=',')

    avg_acc, avg_acc_per_age_cat, avg_acc_per_gender_cat, avg_acc_per_ethnicity_cat, average_bias, b_a, b_g, b_e = evaluate(pretictions_with_gt, full_annotations)

    wandb.log(data={
            'Average accuracy': avg_acc,
            'Avg. acc. age': avg_acc_per_age_cat,
            'Avg. acc. gender': avg_acc_per_gender_cat,
            'Avg. acc. ethnicity': avg_acc_per_ethnicity_cat,
            'Average bias': average_bias,
            'Avg. bias age': b_a,
            'Avg. bias gender': b_g,
            'Avg. bias ethnicity': b_e,
        })


    return train_accuracy / train_loss

