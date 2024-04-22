from transformers import BertModel, BertTokenizer
from torchvision import models
import torch.nn as nn
import torch
import numpy as np
from models.simpleconvnet import SimpleConvNet
from models.imagenet import ImageNet
from utils import AdditiveFusion, TensorFusion


class ImageModel(nn.Module):
    def __init__(self, load_weights):
        super(ImageModel, self).__init__()
        self.base_model = ImageNet()

        self.fc = nn.Linear(2048, 768)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class TextModel(nn.Module):
    def __init__(self, device, model_name='bert-base-uncased'):
        super(TextModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device
    
    def forward(self, captions):
        #Batch encode plus expects batches of sequences (captions = ["Caption 1", "Caption 2", "Caption 3"...])
        encoded_dict = self.tokenizer.batch_encode_plus(
                    captions,
                    add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
                    max_length=30,              # Adjust sentence length (it will add padding for the shorter sentences)                    
                    padding=True,     # Pad/truncate sentences
                    truncation=True,
                    return_attention_mask=True, # Generate attention masks to ignore the padding
                    return_tensors='pt',        # Return PyTorch tensors
                ).to(self.device)
        
        input_ids = encoded_dict['input_ids']  # Token IDs
        attention_mask = encoded_dict['attention_mask']  # Attention mask
        outputs = self.model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  # This contains the embeddings
        word_embeddings = word_embeddings[:,0,:] # We get the embedding of the [CLS] token, which represents the global semantic representation of the whole sentence

        return word_embeddings

class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        # Process the 128-dimensional embeddings from VGGish
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 768)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MultimodalModel(nn.Module):
    def __init__(self, params, device, fusion = 'additive', load_weights = 'True'):
        super(MultimodalModel, self).__init__()
        self.params = params
        self.image_model = ImageModel(load_weights)
        self.audio_model = AudioModel()
        self.text_model = TextModel(device= device)
        self.classifier = nn.Linear(768*3, 7)  # Assume 10 classes 

        if fusion == 'additive':
            self.fusion = AdditiveFusion(3, 768, device)
        elif fusion == 'tensorfusion':
            self.fusion = TensorFusion(num_tensors=1, tensor_dim=(self.params['batch_size'], 768, 768), device=device)

    def forward(self, image, audio, text):
        image_features = self.image_model(image)
        audio_features = self.audio_model(audio)
        text_features = self.text_model(text)
        combined_features = torch.cat((image_features, audio_features, text_features), dim=1)

        output = self.classifier(combined_features)
        return output