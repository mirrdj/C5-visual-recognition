import optuna
import wandb
import os

from torch import nn
import torch

from run_model import run_model
from trainer import train
from validation import validation
from datasets import FIImageAudioDataset, FIImageTextDataset, FIAudioTextDataset, FIImageDataset, FIDataset
from models import ImageAudioModel, ImageTextModel, AudioTextModel, ImageModel, MultimodalModel
from tester import test

id = os.getenv('SLURM_JOB_ID')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_combination(comb):
    if comb == "imgacc":
        return FIImageAudioDataset, ImageAudioModel
    elif comb == "imgtxt":
        return FIImageTextDataset, ImageTextModel
    elif comb == "acctxt":
        return FIAudioTextDataset, AudioTextModel
    elif comb == "img":
        return FIImageDataset, ImageModel
    elif comb == "multi":
        return FIDataset, MultimodalModel
    else:
        raise ValueError(f"{comb} is not a valid combination")

def objective_model_cv(trial):
    params = {
        #'unfrozen_layers': trial.suggest_categorical('unfrozen_layers', ["1"]),  # 1,2,3,4,5
        
        'batch_size': trial.suggest_categorical('batch_size', [16]),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', [224]),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_categorical('lr', [0.0001]),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['sgd']),  # adadelta, adam, sgd, RMSprop
        'momentum': trial.suggest_float('momentum', 0.95, 0.95),

        'rot': trial.suggest_categorical('rot', [20]),
        'sr': trial.suggest_categorical('sr', [0]),
        'zr': trial.suggest_categorical('zr', [0.2]),
        'hf': trial.suggest_categorical('hf', [0.2]),

        'n_filters_1': trial.suggest_categorical('n_filters_1', [16]),
        'n_filters_2': trial.suggest_categorical('n_filters_2', [32]),
        'n_filters_3': trial.suggest_categorical('n_filters_3', [64]),
        'n_filters_4': trial.suggest_categorical('n_filters_4', [128]),

        'kernel_size_1': trial.suggest_categorical('kernel_size_1', [3]),
        'kernel_size_2': trial.suggest_categorical('kernel_size_2', [3]),
        'kernel_size_3': trial.suggest_categorical('kernel_size_3', [5]),
        'kernel_size_4': trial.suggest_categorical('kernel_size_4', [5]),

        'stride': trial.suggest_int('stride', 1, 1),

        'pool': trial.suggest_categorical('pool', ['max']),

        'padding': trial.suggest_categorical('padding', ['same']),
        'neurons': trial.suggest_categorical('neurons', [256]),
        'metadata': trial.suggest_categorical('metadata', [3]),


        'dropout': trial.suggest_categorical('dropout', [0]),
        'bn': trial.suggest_categorical('bn', [True]),
        'L2': trial.suggest_categorical('L2', [False]),
        'epochs': 100,
        'depth': trial.suggest_int('depth', 4, 4),
        'output': 7,

        'resampling': False,

        'task': 'B',
        'save_predictions': True
    }

    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = f'{str(id)}_TASK_{str(params["task"])}_multimodal_concatenate'
    wandb.init(
        project='C5_week6_combined',
        entity='c3_mcv',
        name=execution_name,
        config=config,
        reinit=True,
    )

    class_weights = torch.tensor([600.6, 36.62, 4.75, 2.048, 4.44, 25.89, 117.765]).to(device) # can be changed to use class weights

    # First run was ImageAudio
    # Second run was imgtxt
    # Last acctxt
    comb = 'acctxt'
    dataset, model = get_combination(comb)
    criterion = nn.CrossEntropyLoss(weight=class_weights) 
    model = model(params, device=device)

    ratio = run_model(params, model, train, validation, dataset, criterion, trial.number, id, test, True)

    return ratio


study = optuna.create_study(direction="maximize", study_name='C5-Week6')
study.optimize(objective_model_cv, n_trials=1)