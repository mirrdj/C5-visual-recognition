from torch.nn.functional import pad

# DONE: We add a dimension projection with a linear layer
# Constant padding with 0
def adjust_length_last_dim(tensor, new_len):
    curr_len = tensor.shape[-1]
    delta = new_len - curr_len

    return pad(tensor, pad=(0, delta), mode='constant', value=0)