import torchvision.models as models
import torch.nn as nn

class ImageNet(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, embedding_dim = 1024):
        super(ImageNet, self).__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)
       
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.fc = nn.Linear(num_features, embedding_dim) 


    def forward(self, frame):
        out = self.model(frame)
        batch_size = out.shape[0]
        return out.view(batch_size, -1) #Output is [Batch_size, embedding_size]
