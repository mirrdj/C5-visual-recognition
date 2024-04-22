import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveFusion(nn.Module):
    def __init__(self, num_tensors):
        super(AdditiveFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_tensors))
        self.requires_grad = True

    def forward(self, *tensors):
        norm_weights = F.softmax(self.weights, dim=0)
        fused_tensor = sum(w * t for w, t in zip(norm_weights, tensors))

        return fused_tensor


# fusion_layer = AdditiveFusion(num_tensors=3)

# tensor1 = torch.tensor([1.0, 2.0, 3.0])
# tensor2 = torch.tensor([4.0, 5.0, 6.0])
# tensor3 = torch.tensor([7.0, 8.0, 9.0])

# fused_tensor = fusion_layer(tensor1, tensor2, tensor3)
# print(fused_tensor)
