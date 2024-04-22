from transformers import BertModel, BertTokenizer
from torchvision import models
import torch.nn as nn
from models.simpleconvnet import SimpleConvNet
from models.imagenet import ImageNet
from utils import AdditiveFusion, TensorFusion
from .multimodal import AudioModel, TextModel, ImageModel


class ImageAudioModel(nn.Module):
    def __init__(self, params, device, load_weights = 'True'):
        super(ImageAudioModel, self).__init__()
        self.params = params
        self.image_model = ImageModel(load_weights)
        self.audio_model = AudioModel()
        self.classifier = nn.Linear(768, 7)  # Assume 10 classes 
        self.fusion = AdditiveFusion(3, 768, device)

    def forward(self, image, audio):
        image_features = self.image_model(image)
        audio_features = self.audio_model(audio)
        combined_features = self.fusion(image_features, audio_features)
        output = self.classifier(combined_features)

        return output
    

class ImageTextModel(nn.Module):
    def __init__(self, params, device, load_weights = 'True'):
        super(ImageTextModel, self).__init__()
        self.params = params
        self.image_model = ImageModel(load_weights)
        self.text_model = TextModel(device=device)
        self.classifier = nn.Linear(768, 7)  # Assume 10 classes 
        self.fusion = AdditiveFusion(3, 768, device)

    def forward(self, image, text):
        image_features = self.image_model(image)
        text_features = self.text_model(text)
        combined_features = self.fusion(image_features, text_features)
        output = self.classifier(combined_features)

        return output
    

class AudioTextModel(nn.Module):
    def __init__(self, params, device, load_weights = 'True'):
        super(AudioTextModel, self).__init__()
        self.params = params
        self.text_model = TextModel(device=device)
        self.audio_model = AudioModel()
        self.classifier = nn.Linear(768, 7)  # Assume 10 classes 
        self.fusion = AdditiveFusion(3, 768, device)

    def forward(self, audio, text):
        audio_features = self.audio_model(audio)
        text_features = self.text_model(text)
        combined_features = self.fusion(audio_features, text_features)
        output = self.classifier(combined_features)

        return output