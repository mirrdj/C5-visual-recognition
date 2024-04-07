import torch
import faiss
import yaml
import json
import numpy as np
import torch.nn as nn
from typing import Union, Any

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, top_k_accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils import data_augmentation, get_indices
from models import ImageNet, TorchTextNet, TorchTextNetBert

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

# def retrieval_itt(img, image_model, text_index):
#     # Extract features using the model
#     features = image_model(img.unsqueeze(0))
#     features = features.squeeze().cpu().detach().numpy()

#     distances, indices = text_index.search(features[None, ...], k=text_index.ntotal)
#     return distances, indices, features

def evaluate_retrieval_itt(text_encoder, database_path: str, image_model: Union[str, nn.Module], text_model: Union[str, Path, nn.Module], embedding_size: int, config: Any):
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
    #Create text index
    index = faiss.IndexFlatL2(embedding_size)
    txt_feats_list = []
    txt_ids = []
    for entry in tqdm(database["annotations"]):
        txt_ids.append(int(entry["image_id"]))
        with torch.no_grad():
            # embedding = text_model([entry["caption"]])
            # print(entry["caption"])
            # print(get_indices(vocab, entry["caption"]))
            if text_encoder == 'FASTTEXT':
                embedding = text_model([get_indices(vocab, entry["caption"]).to(device)])
            else:
                embedding = text_model([entry["caption"]])

            if not embedding.is_contiguous():
                embedding = embedding.contiguous()
            # print(embedding.shape)
            txt_feats_list.append(embedding.squeeze().cpu().detach().numpy())
            # index.add([embedding.squeeze().cpu().detach().numpy()])
    
    sum_ap = 0
    sum_precission_at1 = 0
    sum_precission_at5 = 0
    sum_accuracy_at5 = 0
    # Evaluate retrieval for every query
    img_feats_list = []
    img_ids= []

    for entry in tqdm(database["images"]):
        filename = entry["file_name"]
        image_id = entry["id"]

        if "train" in filename:
            img_path = Path(config["DATASET_DIR"]) / config["TRAIN_FOLDER"] / filename
        elif "val" in filename:
            img_path = Path(config["DATASET_DIR"]) / config["VAL_FOLDER"] / filename
        img = Image.open(img_path).convert('RGB')
        img = transform(img).to(device)

        features = image_model(img.unsqueeze(0))
        features = features.squeeze().cpu().detach().numpy()
        img_feats_list.append(features)
        img_ids.append(image_id)

    txt_feats_list = np.array(txt_feats_list)
    img_feats_list = np.array(img_feats_list)
    # faiss.normalize_L2(txt_feats_list)
    # faiss.normalize_L2(img_feats_list)
    index.add(txt_feats_list)

    # for entry in tqdm(database["images"]):
    for feats, image_id in tqdm(zip(img_feats_list, img_ids)):
        # filename = entry["file_name"]
        # image_id = entry["id"]

        # if "train" in filename:
        #     img_path = Path(config["DATASET_DIR"]) / config["TRAIN_FOLDER"] / filename
        # elif "val" in filename:
        #     img_path = Path(config["DATASET_DIR"]) / config["VAL_FOLDER"] / filename
        # img = Image.open(img_path).convert('RGB')
        # img = transform(img).to(device)

        distances, indices = index.search(feats[None, ...], k=index.ntotal)
        # img_feats_list.append(feats)
        true_bin = [1 if database["annotations"][i]["image_id"] == image_id else 0 for i in indices[0]]
        # scores = (distances.max() - distances[0])/distances.max()
        # ap = average_precision_score(true_bin, scores)
        AP, precission, recall = compute_ap(true_bin, np.sum(np.array(true_bin) == 1))
        # precission, recall, thresholds = precision_recall_curve(true_bin, scores)
        # print(true_bin)
        # print(precission, scores)
        # accuracy = top_k_accuracy_score(true_bin, scores, k=5)
        print(f'IMG: {image_id} -> TOP 1 CAPTION (IMG {database["annotations"][indices[0][0]]["image_id"]}, {database["annotations"][indices[0][1]]["image_id"]}, {database["annotations"][indices[0][2]]["image_id"]}, {database["annotations"][indices[0][3]]["image_id"]}, {database["annotations"][indices[0][4]]["image_id"]}): {database["annotations"][indices[0][0]]["caption"]}')
        
        print(f'IMG: {image_id} -> TOP 1 CAPTION (IMG {database["annotations"][indices[0][0]]["image_id"]}, {database["annotations"][indices[0][1]]["image_id"]}, {database["annotations"][indices[0][2]]["image_id"]}, {database["annotations"][indices[0][3]]["image_id"]}, {database["annotations"][indices[0][4]]["image_id"]}): P@1: {precission[0]}, P@5: {precission[4]}, AP: {AP}')

        sum_ap += AP
        sum_precission_at1 += precission[0]
        sum_precission_at5 += precission[4]
        # sum_accuracy_at5 += accuracy
    #TODO: Plot precision-recall?
    #TODO: Extract visualization
    n_queries = len(database["images"])
    sum_ap /= n_queries
    sum_precission_at1 /= n_queries
    sum_precission_at5 /= n_queries
    # sum_accuracy_at5 /= n_queries
    print(f'AP: {sum_ap}, P@1: {sum_precission_at1}, P@5: {sum_precission_at5}')

    #Visualization
    tsne = TSNE(n_components=2, random_state=0, verbose=1, metric='euclidean')
    X = np.vstack([np.array(txt_feats_list), np.array(img_feats_list)])
    y = np.hstack([np.zeros(len(database["annotations"])), np.ones(len(database["images"]))])
    image_ids = txt_ids+img_ids
    diff1 = set(txt_ids) - set(img_ids)
    print(diff1)
    # umap_visualization(X, y)
    X_embedded = tsne.fit_transform(X)

    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], c='r', label='Text embeddings' , s=10)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], c='b', label='Image embeddings', s=10)


    random_indices = np.random.choice(np.where(y == 0)[0], size=60, replace=False)
    random_indices_2 = np.random.choice(np.where(y == 1)[0], size=20, replace=False)
    # for i, (x, y, image_id) in enumerate(zip(X_embedded[y == 0, 0][:30], X_embedded[y == 0, 1][:30], np.array(image_ids)[y == 0][:30])):
    #     plt.annotate(image_id, (x, y), fontsize=6)

    for i in random_indices:
        plt.annotate(image_ids[i], (X_embedded[i, 0], X_embedded[i, 1]), fontsize=8)  # Adjust the fontsize as needed
    
    for i in random_indices_2:
        plt.annotate(image_ids[i], (X_embedded[i, 0], X_embedded[i, 1]), fontsize=8)  # Adjust the fontsize as needed

    plt.title('t-SNE Visualization of text and images embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # Save the plot as a PNG file
    plt.savefig(f'TSNE_plot/tsne_plot_TaskA.png')

    return sum_ap, sum_precission_at1, sum_precission_at5

if __name__ == "__main__":
    with open('/ghome/group02/C5-G2/Week4/config.yml') as file:
        config = yaml.safe_load(file)
    
    dataset_root = Path(config['DATASET_DIR'])
    dataset_train_file = config['TRAIN_FILE']
    dataset_val_file = config['VAL_FILE']

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

    evaluate_retrieval_itt('BERT','/ghome/group02/C5-G2/Week4/captions_test_100.json', '/ghome/group02/C5-G2/Week4/weights/textimagenet35212_0_img.pth', '/ghome/group02/C5-G2/Week4/weights/textimagenet35212_0_txt.pth', 2048, config)
    # evaluate_retrieval_itt('FASTTEXT','/ghome/group02/C5-G2/Week4/captions_test_100.json', '/ghome/group02/C5-G2/Week4/weights/textimagenet35242_0_img.pth', '/ghome/group02/C5-G2/Week4/weights/textimagenet35242_0_txt.pth', 2048, config)