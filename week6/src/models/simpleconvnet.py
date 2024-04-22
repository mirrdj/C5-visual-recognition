import torch
import torch.nn as nn


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='valid', pooling='max', batch_norm=True, dropout=0, pool = 'max'):
    super(ConvBlock, self).__init__()

    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        
    if batch_norm:
      layers.append(nn.BatchNorm2d(out_channels))  # Batch normalization
    
    layers.append(nn.ReLU())  # ReLU activation function
    
    if dropout!=0:
      layers.append(nn.Dropout(dropout)) #Dropout
    
    if pool == 'max':
      layers.append(nn.MaxPool2d(2, 2))  # Max pooling
    elif 'pool' == 'avg':
      layers.append(nn.AvgPool2d(2, 2))  #Avg pooling
        
    self.conv_block = nn.Sequential(*layers)
        
  def forward(self, x):
    return self.conv_block(x)


class SimpleConvNet(nn.Module):
  def __init__(self, params):
    super(SimpleConvNet, self).__init__()
    self.depth = params['depth']
    if self.depth == 2:
        output_size = int(params['n_filters_2'])
    elif self.depth == 3:
        output_size = int(params['n_filters_3'])
    elif self.depth == 4:
        output_size = int(params['n_filters_4'])
    else:
      raise ValueError("Invalid depth")
    
    self.conv_block1 = ConvBlock(in_channels=3, out_channels=int(params['n_filters_1']), kernel_size=int(params['kernel_size_1']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']), pool=params["pool"])
    self.conv_block2 = ConvBlock(in_channels=int(params['n_filters_1']), out_channels=int(params['n_filters_2']), kernel_size=int(params['kernel_size_2']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']), pool=params["pool"])
    self.conv_block3 = ConvBlock(in_channels=int(params['n_filters_2']), out_channels=int(params['n_filters_3']), kernel_size=int(params['kernel_size_3']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']), pool=params["pool"])        
    self.conv_block4 = ConvBlock(in_channels=int(params['n_filters_3']), out_channels=int(params['n_filters_4']), kernel_size=int(params['kernel_size_4']), stride= params['stride'], padding=params['padding'], batch_norm=params['bn'], dropout=float(params['dropout']), pool=params["pool"])    

    self.globavgpool = nn.AdaptiveMaxPool2d((1, 1)) #Global avg pooling
    self.fc = nn.Linear(output_size, int(params['neurons']))
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(int(params['neurons']), params['output'])
        
  def forward(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    if self.depth > 2:
      x = self.conv_block3(x)
      if self.depth > 3:
        x = self.conv_block4(x)

    x = self.globavgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x