import torch
import faiss
import yaml
import json
import numpy as np
import torch.nn as nn
import os
import pickle

from typing import Union, Any
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, top_k_accuracy_score
from sklearn.manifold import TSNE
# import umap.umap_ as umap
import matplotlib.pyplot as plt

from utils import data_augmentation, get_indices
from models import ImageNet, TorchTextNet, TorchTextNetBert

# def umap_visualization(embed, y):
#     print(type(embed))
#     embed = embed.astype(object)
#     print(type(embed))
#     standard_embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(embed)

#     plt.scatter(standard_embedding[y == 0, 0], standard_embedding[y == 0, 1], c='r', label='Text embeddings')
#     plt.scatter(standard_embedding[y == 1, 0], standard_embedding[y == 1, 1], c='b', label='Image embeddings')
#     plt.title('t-SNE Visualization of text and images embeddings')
#     plt.xlabel('t-SNE Dimension 1')
#     plt.ylabel('t-SNE Dimension 2')
#     plt.legend()
#     plt.grid(True)
#     # plt.show()

#     # Save the plot as a PNG file
#     plt.savefig(f'TSNE_plot/tsne_plot_TaskB_UMAP.png')

def compute_ap(true_bin, num_img_class_gt):

  precision = []
  recall = []
  AP = 0
  num = 0

  for i, prediction in enumerate(true_bin):

      if prediction == 1:
          num +=1
          AP += (num/(i+1))
      
      precision.append(num/(i+1))
      recall.append(num/num_img_class_gt)
          
  AP = AP/num

  return AP, precision, recall

def retrieval_tti(caption, text_model, img_index):
    # Extract features using the model
    features = text_model([caption])
    # print(features.shape)
    features = features.squeeze().cpu().detach().numpy()
    # print(features.shape)

    distances, indices = img_index.search(features[None, ...], k=img_index.ntotal)
    return distances, indices, features

def evaluate_retrieval_tti(text_encoder, database_path: str, image_model: Union[str, nn.Module], text_model: Union[str, Path, nn.Module], embedding_size: int, config: Any, caption):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if CUDA is available, and set map_location accordingly
    map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Load image and text models
    if not isinstance(image_model, nn.Module):
        img_model = ImageNet(embedding_dim= embedding_size)
        img_model.load_state_dict(torch.load(image_model, map_location=map_location))
        img_model.to(device)
        image_model = img_model
    image_model.eval()

    if not isinstance(text_model, nn.Module):
        if text_encoder == 'FASTTEXT':
            txt_model = TorchTextNet(embedding_size)
            vocab = txt_model.get_vocab()
        else:
            txt_model = TorchTextNetBert(embedding_dim=embedding_size, device=device)
        
        txt_model.load_state_dict(torch.load(text_model, map_location=map_location))
        txt_model.to(device)
        text_model = txt_model
    text_model.eval()

    transform = data_augmentation(False)

    with open(database_path) as f:
        database = json.load(f)
    
    #Extract database (text) features, project to image space
    # Create text index
    index = faiss.IndexFlatL2(embedding_size)
    img_feats_list = []
    img_ids = []
    for entry in tqdm(database["images"]):
        filename = entry["file_name"]
        image_id = entry["id"]
        img_ids.append(int(image_id))

        if "train" in filename:
            img_path = Path(config["DATASET_DIR"]) / config["TRAIN_FOLDER"] / filename
        elif "val" in filename:
            img_path = Path(config["DATASET_DIR"]) / config["VAL_FOLDER"] / filename
        img = Image.open(img_path).convert('RGB')
        img = transform(img).to(device)
        with torch.no_grad():
            embedding = image_model(img.unsqueeze(0))
            img_feats_list.append(embedding.squeeze().cpu().detach().numpy())
    img_feats_list = np.array(img_feats_list)

    with open('feats_validation_approach1_whole_dataset_approach0.pkl', 'wb') as f:  # Note the 'wb' mode for binary writing
        pickle.dump(img_feats_list, f)
    # faiss.normalize_L2(img_feats_list)
    # faiss.normalize_L2(feats_query)
    index.add(img_feats_list)
    print(index.ntotal)
    
    sum_ap = 0
    sum_precission_at1 = 0
    sum_precission_at5 = 0
    sum_accuracy_at5 = 0
    txt_feats_list = []
    txt_ids = []

    # Get image retrieval for given query
    distances, indices, feats = retrieval_tti(caption, text_model, index)

    print(f'Caption (IMG {image_id}): {caption} -> TOP 5 IMG: {database["images"][indices[0][0]]["id"]},{database["images"][indices[0][1]]["id"]}, {database["images"][indices[0][2]]["id"]}, {database["images"][indices[0][3]]["id"]}, {database["images"][indices[0][4]]["id"]}')
    for i, idx in enumerate(indices[0][:100]):
        
        img_id = [database["images"][idx]["id"]]
        for entry in database["annotations"]:
            if entry["image_id"] in img_id:
                caption = entry['caption']
                continue
        img_path = Path(config["DATASET_DIR"]) / config["VAL_FOLDER"] / database["images"][idx]["file_name"]
        img = Image.open(img_path).convert('RGB')
        img.save(f"retrieved_imgs/{i}_{caption}.jpg")

    return 

if __name__ == "__main__":
    with open('/ghome/group02/C5-G2/Week4/config.yml') as file:
        config = yaml.safe_load(file)
    
    dataset_root = Path(config['DATASET_DIR'])
    dataset_train_file = config['TRAIN_FILE']
    dataset_val_file = config['VAL_FILE']

    caption = 'Groom and bride'
    output_path = 'retrieved_imgs'

    if os.path.exists(output_path):
        os.system('rm -rf "{}"'.format(output_path))
    
    os.makedirs(output_path)


    #with open(Path(dataset_root) / "captions_val2014.json") as f:
    #    queries = json.load(f)
    #print(len(queries))
    #print(type(queries))
    #print(queries.keys())
    #print(queries["images"][0])
    #print(queries["annotations"][0])
    #img_id = queries["annotations"][0]["image_id"]
    #print(next((index for (index, d) in enumerate(queries["images"]) if d["id"] == img_id), None))
    #print(queries["images"][30770])

    evaluate_retrieval_tti('BERT','/ghome/group02/mcv/datasets/C5/COCO/captions_val2014.json', '/ghome/group02/C5-G2/Week5/weights/textimagenet36570_0_img_aux.pth', '/ghome/group02/C5-G2/Week5/weights/textimagenet36570_0_txt_aux.pth', 2048, config, caption)
    # evaluate_retrieval_tti('FASTTEXT','/ghome/group02/C5-G2/Week4/captions_test_100.json', '/ghome/group02/C5-G2/Week4/weights/textimagenet35390_0_img.pth', '/ghome/group02/C5-G2/Week4/weights/textimagenet35390_0_txt.pth', 2048, config)