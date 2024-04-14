
import optuna
import wandb
import similarity as sm
import os

from run_model import run_model
from trainer import  TripletTrainer
from validation import TripletValidator
from models import TextImageNet, ImageNet, TorchTextNet, ImageTextNet, TorchTextNetBert
from datasets.datasets import TripletCOCO_B, TripletCOCO_B_full
from losses import ContrastiveLoss, TripletLoss
import torch

id = os.getenv('SLURM_JOB_ID')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective_model_cv(trial):
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [32]),  # 8,16,32,64
        'img_size': trial.suggest_categorical('img_size', [224]),  # 8,16,32,64,128,224,256
        'lr': trial.suggest_categorical('lr', [0.0001]),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['sgd']),  # adadelta, adam, sgd, RMSprop
        'unfroze': trial.suggest_categorical('unfroze', [20]),

        'rot': trial.suggest_categorical('rot', [20]),
        'sr': trial.suggest_categorical('sr', [0]),
        'zr': trial.suggest_categorical('zr', [0.2]),
        'hf': trial.suggest_categorical('hf', [0.2]),

        'margin': trial.suggest_float('margin', 1.0, 1.0),

        'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        'dropout': trial.suggest_categorical('dropout', ['0']),
        'epochs': trial.suggest_int('epochs', 100, 100),
        'output': trial.suggest_categorical('output', [2048]),

        'dataset_ratio': trial.suggest_float('dataset_ratio', 1.,1.),
        'dataset': trial.suggest_categorical('dataset', ['generated']),
        'num_cap': trial.suggest_int('num_cap', 5,5), #Number of captions per image

        'task': trial.suggest_categorical('task', ['B']), #Task A or B
        'text_encoder': trial.suggest_categorical('text_encoder', ['BERT']), # FASTTEXT / BERT
        'load_weights': trial.suggest_categorical('load_weights', ['True']),
        'freeze': trial.suggest_float('freeze', 0., 0.), #Percentage of layers to be frozen
    }

    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = f'{str(id)}_TASK_{str(params["task"])}_finetune_0_frozen'
    wandb.init(
        project='C5_w5_finetune',
        entity='c3_mcv',
        name=execution_name,
        config=config,
        reinit=True,
    )

    if params['task'] == 'B':
        
        if params['text_encoder'] == 'FASTTEXT':
            model = TextImageNet(ImageNet(embedding_dim=params['output']), TorchTextNet(embedding_dim=params['output']), sm.euclidean)
        elif params['text_encoder'] == 'BERT':
            model = TextImageNet(ImageNet(embedding_dim=params['output']), TorchTextNetBert(embedding_dim=params['output'], device=device), sm.euclidean, load_weights = params['load_weights'], freeze = params['freeze'])
        
        if params['dataset'] == 'generated':
            dataset = TripletCOCO_B
        else:
            dataset = TripletCOCO_B_full
    
    # elif params['task'] == 'A':
        
    #     if params['text_encoder'] == 'FASTTEXT':
    #         model = ImageTextNet(ImageNet(embedding_dim=params['output']), TorchTextNet(embedding_dim=params['output']), sm.euclidean)
    #     elif params['text_encoder'] == 'BERT':
    #         model = ImageTextNet(ImageNet(embedding_dim=params['output']), TorchTextNetBert(embedding_dim=params['output'], device=device), sm.euclidean)
        
    #     dataset = TripletCOCO_A

    trainer = TripletTrainer()
    validator = TripletValidator()
    criterion = TripletLoss(margin=params['margin'])

    ratio = run_model(params, model, trainer, validator, dataset, criterion, trial.number, id)

    return ratio


study = optuna.create_study(direction="maximize", study_name='C5-Week1')
study.optimize(objective_model_cv, n_trials=1)