#from __future__ import print_function
import os
import torch
import json
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

def choose_multiple(options):
    combinations = []
    for r in range(1, len(options) + 1):
        combinations.extend([tuple(x) for x in itertools.combinations(iterable=options, r=r)])

    return combinations
    


def get_optimizer(params, model):
    if params['optimizer'] == 'adam':
        #optimizer = Adam(learning_rate=float(params['lr']), beta_1=float(params['momentum']))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params['lr']))
    elif params['optimizer'] == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=float(params['lr']), rho=float(params['momentum']))
    elif params['optimizer'] == 'sgd':
        #optimizer = SGD(learning_rate=float(params['lr']), momentum=float(params['momentum']))
        optimizer = torch.optim.SGD(model.parameters(), lr = float(params['lr']))
    elif params['optimizer'] == 'RMSprop':
        #optimizer = RMSprop(learning_rate=float(params['lr']), rho=float(params['momentum']))
        optimizer = torch.optim.RMSprop(model.parameters(), lr=float(params['lr']))
    else:
        raise ValueError(f"No optimizer: {params['optimizer']}")
        

    return optimizer


def json_writer(data, path, write=False):
    """
    Read and write DB jsons
    """
    if write:
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            return data


def get_imgs_paths(path, split):
    imgs_list = []
    for folder in os.listdir(os.path.join(path, split)):
        for img in os.listdir(os.path.join(path, split, folder)):
            imgs_list.append(os.path.join(path, split, folder, img))

    imgs_list = sorted(imgs_list)
    return imgs_list

def get_imgs_lbls_dict(annotations):
    image_labels = dict()
    for key, value in annotations.items():
        for image_id in value:
            if image_id not in image_labels:
                image_labels[image_id] = []
            image_labels[image_id].append(int(key))
    return image_labels

def reorder_csv(predicted_csv, split='test'):
    
    # Read the first CSV file
    with open(predicted_csv, 'r') as file1:
        reader1 = csv.reader(file1)
        data1 = list(reader1)
    
    # Read the second CSV file
    with open('/ghome/group02/C5-G2/Week6/src/evaluation_script/predictions_test_set.csv', 'r') as file2:
        reader2 = csv.reader(file2)
        data2 = list(reader2)

    # Extract the values from the first column of the second file
    second_column_values = [row[0] for row in data2]

    # Sort the data from the first file based on the order of the first column in the second file
    data1_sorted = sorted(data1, key=lambda x: second_column_values.index(x[0]))

    # Write the sorted data back to the first CSV file
    with open(predicted_csv, 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        writer1.writerows(data1_sorted)

def my_test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device, params, reorder=True):

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
        for data in tqdm(dataloader, desc="Testing"):
            # Send data to target device
            names, X, y = data

            if(len(y)!=1):
                print("ERROR: batch size (test stage) =", len(y))
                print("ERROR: my_test_step function require batch size = 1 to generate the correct output file")
                exit()
            
            X = X.to(device)
            y = y.to(device)
            
            # 1. Forward pass
            y_pred = model(X)

            # Calculate and accumulate accuracy metric across all batches
            _, predicted = torch.max(y_pred, 1)
            test_acc += (predicted == y).sum().item()

            if params["save_predictions"]:
                with open(preds_file, "a") as f:
                    for i, name in enumerate(names):
                        f.write(f"{name},{dataloader.dataset.CLASSES[y[i]]},{dataloader.dataset.CLASSES[predicted[i]]}\n")

            counter +=1

    if reorder:
        reorder_csv(preds_file)

    # Adjust metrics to get average accuracy per batch 
    test_acc /= len(dataloader.dataset)
    print("Average accuracy = ", test_acc)


def my_test_step_MM(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device, params):

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
        for img, audio, text, labels in tqdm(dataloader):
            # Send data to target device

            names, _, age_group, _, _ = labels
            
            img = img.to(device)
            audio = audio.to(device)
            age_group = age_group.to(device)

            if(len(img)!=1):
                print("ERROR: batch size (test stage) =", len(img))
                print("ERROR: my_test_step function require batch size = 1 to generate the correct output file")
                exit()
            
            
            # 1. Forward pass
            y_pred = model(img, audio, text)

            # Calculate and accumulate accuracy metric across all batches
            _, predicted = torch.max(y_pred, 1)
            test_acc += (predicted == age_group).sum().item()

            if params["save_predictions"]:
                with open(preds_file, "a") as f:
                    for i, name in enumerate(names):
                        f.write(f"{name},{dataloader.dataset.CLASSES[age_group[i]]},{dataloader.dataset.CLASSES[predicted[i]]}\n")

            counter +=1

    reorder_csv(preds_file)

    # Adjust metrics to get average accuracy per batch 
    test_acc /= len(dataloader.dataset)
    print("Average accuracy = ", test_acc)


