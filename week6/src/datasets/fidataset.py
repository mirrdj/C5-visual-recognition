import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor
from torchvision.io import read_image
import os
import pickle
import numpy as np



from utils import read_csv

# Used as a base class for other datasets
# Not ideal but it makes dev cleaner
class FIDataset(Dataset):
        CLASSES = ["1", "2", "3", "4", "5", "6", "7"]

        def __init__(self, data_dir='./data/First_Impressions_v3_multimodal', split='train'):
            self.transform = T.Compose([T.ConvertImageDtype(torch.float32),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            vis_ext = 'jpg'
            txt_ext = 'pkl'
            acc_ext = '_audio.pkl'

            self.names = [] # User ids + index
            self.visual = [] # Path to images
            self.text = [] # Path to text
            self.accoustic = [] # Path to accoustic
            self.age = [] # Label according to the strategy used in fine-tuning

            # Read file containing age, gender, ethnicity
            label_file = f"{split}_set_age_labels.csv"
            rows = read_csv(data_dir, label_file)

            for row in rows[1:]:
                video_name, _, age_group, _, _ = row
                image_path = f'{data_dir}/{split}/{age_group}/{video_name[:-4]}.{vis_ext}'
                text_path = f'{data_dir}/{split}/{age_group}/{video_name[:-4]}.{txt_ext}'
                audio_path = f'{data_dir}/{split}/{age_group}/{video_name[:-4]}_audio.pkl'

                self.names.append(video_name)
                self.visual.append(image_path)
                self.text.append(text_path)
                self.accoustic.append(audio_path)
                self.age.append(self.CLASSES.index(row[2]))

        def _read_image(self, img_path):
            img = read_image(img_path)
            img = self.transform(img)

            return img

        def _read_text(self, txt_path):  
            with open(txt_path, 'rb') as text_file:
                text = pickle.load(text_file)
            
            return text

        def _read_accoustic(self, acc_path):
            with open(acc_path, 'rb') as audio_file:
                audio = pickle.load(audio_file)
            audio = np.mean(audio, axis=0)

            return audio


        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            # Every __getitem__ in subclass should respect this order in return
            # name, vis, accoustic, text, age

            name = self.names[idx]             
            vis = self._read_image(self.visual[idx])
            accoustic = self._read_accoustic(self.accoustic[idx])
            text = self._read_text(self.text[idx]) 
            age = self.age[idx]


            return (name, vis, accoustic, text, age)


class FIImageDataset(FIDataset):
    def __getitem__(self, idx):
        name = self.names[idx]             
        vis = self._read_image(self.visual[idx])
        age = self.age[idx]

        return (name, vis, age)


class FIImageAudioDataset(FIDataset):
    def __getitem__(self, idx):
        name = self.names[idx]             
        vis = self._read_image(self.visual[idx])
        accoustic = self._read_accoustic(self.accoustic[idx])
        age = self.age[idx]

        return (name, vis, accoustic, age)


class FIImageTextDataset(FIDataset):
    def __getitem__(self, idx):
        name = self.names[idx]             
        vis = self._read_image(self.visual[idx])
        text = self._read_text(self.text[idx]) 
        age = self.age[idx]

        return (name, vis, text, age)


class FIAudioTextDataset(FIDataset):
    def __getitem__(self, idx):
        name = self.names[idx]             
        accoustic = self._read_accoustic(self.accoustic[idx])
        text = self._read_text(self.text[idx]) 
        age = self.age[idx]

        return (name, accoustic, text, age)

