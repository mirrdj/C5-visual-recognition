import torchvision.models as models
import torch.nn as nn
import torch

class ImageNet(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, embedding_dim = 7):
        super(ImageNet, self).__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)
       
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.fc = nn.Linear(num_features, embedding_dim) 

        # Load the state dictionary
        state_dict = torch.load('/ghome/group02/C5-G2/Week6/weights/ImageNet_38850_0')

        # Remove the 'model' prefix from the keys in the state dictionary
        for key in list(state_dict.keys()):
            if 'model.' in key:
                state_dict[key.replace('model.', '')] = state_dict.pop(key)
        
        self.model.load_state_dict(state_dict)

        self.model = nn.Sequential(*(list(self.model.children())[:-1]))


    def forward(self, frame):
        out = self.model(frame)
        # print(out.shape)
        batch_size = out.shape[0]
        return out.view(batch_size, -1) #Output is [Batch_size, embedding_size]
