from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TensorFusion(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, tensor_dim=(1, 768, 768), num_tensors=1, device = 'cuda'):
        '''
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super().__init__()

        self.weight_vecs = [LearnableVec(tensor_dim, device=device) for _ in range(num_tensors)]
        self.weight.requires_grad = True

    def forward(self, img_emb, text_emb, audio_emb):
        '''
        Args:
            img_emb: tensor of shape (batch_size, img_len)
            text_emb: tensor of shape (batch_size, text_len)
            audio_emb: tensor of shape (batch_size, audio_len)
        '''
        # norm_weight = F.softmax(self.weight, dim=0)
        batch_size = audio_emb.data.shape[0]

        fusion_tensor = torch.bmm(audio_emb.unsqueeze(2), img_emb.unsqueeze(1))
        
        fusion_tensor = fusion_tensor.view(-1, 1)
        fusion_tensor = torch.bmm(fusion_tensor, text_emb.unsqueeze(1)).view(batch_size, -1)


        return self.weight_vecs * fusion_tensor
    

class LearnableVec(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.device=device
        self.W = torch.nn.Parameter(torch.randn(n)).to(self.device)
        # self.W.requires_grad = True
  
    def forward(self, x):
        # self.W = self.W.
        return self.W * x


class AdditiveFusion(nn.Module):
    def __init__(self, num_vecs, vec_dim, device) -> None:
        super().__init__()
        self.weight_vecs = [LearnableVec(vec_dim, device) for i in range(num_vecs)]
    
    def forward(self, *x):
        res = self.weight_vecs[0](x[0])
        for i in range(1, len(x)):
            res += self.weight_vecs[i](x[i])
        return res