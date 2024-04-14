import torch.nn as nn
from torchvision import models
import torch
from utils import adjust_length_last_dim as adjust_length
import time

class TripletNet(nn.Module):
    def __init__(self, image_net, text_net, similarity_method, load_weights = 'False', freeze = 0):
        super(TripletNet, self).__init__()
        self.image_net = image_net
        self.text_net = text_net
        if load_weights == 'True':
            self.image_net.load_state_dict(torch.load('/ghome/group02/C5-G2/Week4/weights/textimagenet35263_0_img.pth'))
            self.text_net.load_state_dict(torch.load('/ghome/group02/C5-G2/Week4/weights/textimagenet35263_0_txt.pth'))
        self.similarity_method = similarity_method

        if freeze != 0.:
            # Get the total number of layers in ResNet50
            total_layers = len(list(self.image_net.children()))

            # Calculate the number of layers to freeze (75% of the total)
            num_layers_to_freeze = int(freeze * total_layers)

            # Freeze the layers
            for idx, child in enumerate(self.image_net.children()):
                if idx < num_layers_to_freeze:
                    for param in child.parameters():
                        param.requires_grad = False


            # Get the total number of layers in BERT
            total_layers = len(self.text_net.model.encoder.layer)

            # Calculate the number of layers to freeze (75% of the total)
            num_layers_to_freeze = int(freeze * total_layers)

            # Freeze the layers
            for layer in self.text_net.model.encoder.layer[:num_layers_to_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x1, x2, x3):
        pass # This is overwritten by the subclasses

    def adjust_length(self, feature, newlen):
        return adjust_length(feature, newlen)

    def get_vocab(self):
        return self.text_net.get_vocab()

    def state_dict(self):
        img_dict = self.image_net.state_dict()
        txt_dict = self.text_net.state_dict()

        return [img_dict, txt_dict]

    
    def load_state_dict(self, load_txt, load_img):
        self.image_net.load_state_dict(load_img)

# Task A
class ImageTextNet(TripletNet):
    def forward(self, x1, x2, x3):
        output1 = self.image_net(x1)
        output2 = self.text_net(x2)
        output3 = self.text_net(x3)

        return output1, output2, output3

# Task B
class TextImageNet(TripletNet):
    def forward(self, x1, x2, x3):

        output1 = self.text_net(x1)
        output2 = self.image_net(x2)
        output3 = self.image_net(x3)

        return output1, output2, output3

