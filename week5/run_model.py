# from models import TextImageNet, ImageNet
from utils import get_optimizer
# import similarity as sm
# from datasets.datasets_IKER import TripletCOCO_A
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from datasets.img_processing import data_augmentation
from retrieval_tti import evaluate_retrieval_tti
import numpy as np
import yaml
import json
import os
import wandb
import csv


# Set the random seed for Python and NumPy
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device selected: {device}")

with open('config.yml', 'r') as file:
    data = yaml.safe_load(file)

# PATH = data['DATASET_DIR']
# TRAIN_FILE = data['TRAIN_FILE']
# TRAIN_FOLDER = data['TRAIN_FOLDER']
# VAL = data['VAL_FILE']

path_train_file = '/ghome/group02/C5-G2/Week5/Results_2/generated.csv'
path_COCO_train_file = '/ghome/group02/mcv/datasets/C5/COCO/captions_train2014.json'
path_train_folder = '/ghome/group02/C5-G2/Week5/Results_2/XL'
path_COCO_train_folder = '/ghome/group02/mcv/datasets/C5/COCO/train2014'
# path_val = os.path.join(PATH, VAL)

# def run_model(params, model, trainer, validator, dataset, criterion, trial_number, id):
def run_model(params, model, trainer, validator, dataset, criterion, trial_number, id):
    BEST_MODEL_FNAME = './weights/textimagenet'
    model_name_txt =  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '_txt.pth'
    model_name_img =  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '_img.pth'
    model_name_img_aux =  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '_img_aux.pth'
    model_name_txt_aux=  BEST_MODEL_FNAME + str(id) + '_' + str(trial_number) + '_txt_aux.pth'

    model.to(device)
    
    IMG_WIDTH = params['img_size']
    IMG_HEIGHT = params['img_size']
    NUMBER_OF_EPOCHS = params['epochs']
    BATCH_SIZE = params['batch_size']

    with open(path_COCO_train_file, 'r') as f:
        loaded_object = json.load(f)

    gen_dict = {}
    with open(path_train_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            img_id, caption = row[0].split(':')

            if img_id not in gen_dict:
                gen_dict[img_id] = caption

    if params['dataset'] == 'generated':
        dataset_train = dataset(data_dir=path_train_folder, gen_dict= gen_dict, mode='train', transform=data_augmentation(False))
    else:
        dataset_train = dataset(captions = loaded_object, data_dir_COCO = path_COCO_train_folder, data_dir_GEN=path_train_folder, gen_dict= gen_dict, mode='train', transform=data_augmentation(False))

    #Get a split of the dataset for training
    dataset_size = len(dataset_train)
    print('DATASET_SIZE: ', dataset_size)
    print(params['dataset_ratio'])
    train_size = int(dataset_size * params['dataset_ratio'])

    print('DATASET SELECTED TRAINING SIZE: ', train_size)
    aux = dataset_size - train_size

    dataset_train, _ = random_split(dataset_train, [train_size, aux])

    dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True, num_workers=4)

    optimizer = get_optimizer(params, model)

    # Define early stopping parameters
    patience = 5
    # min_delta = 0.001
    best = np.Inf
    current_patience = 0

    for epoch in range(NUMBER_OF_EPOCHS):
        train_loss, train_accuracy = trainer.train(model, dataloader_train, criterion, optimizer, params, device)
        # Adjust learning rate based on validation loss
        # scheduler.step(val_loss)
        torch.save(model.state_dict()[0], model_name_img_aux) #Save resnet weights
        torch.save(model.state_dict()[1], model_name_txt_aux) #Save fasttext + embedding layer + fc weigths

        ap, precission_at1, precission_at5 = evaluate_retrieval_tti(params['text_encoder'],'/ghome/group02/C5-G2/Week4/captions_valsplit_1100.json', model_name_img_aux, model_name_txt_aux, params['output'], '/ghome/group02/C5-G2/Week4/config.yml')
        # Early stopping
        if train_loss < best:
            best = train_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            
            torch.save(model.state_dict()[0], model_name_img) #Save resnet weights
            torch.save(model.state_dict()[1], model_name_txt) #Save fasttext + embedding layer + fc weigths
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{NUMBER_OF_EPOCHS}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f}')
        print(f'Epoch [{epoch+1}/{NUMBER_OF_EPOCHS}], AP|P@1|P@5: {ap:.4f}|{precission_at1:.4f}|{precission_at5:.4f}')

        wandb.log({
            'Train Loss': train_loss,
            'Train Accuracy': train_accuracy,
            'AP': ap,
            'Precision at 1': precission_at1,
            'Precision at 5': precission_at5, 
        })
