#from __future__ import print_function
import os
import torch
import json
import itertools

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