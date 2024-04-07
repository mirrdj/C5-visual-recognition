import re
import torch

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    cleaned_text = re.sub(punctuation_pattern, '', text)
    
    return cleaned_text

def get_indices(vocab, caption):
    caption = remove_punctuation(caption)
    print(caption)
    words = [word.lower() for word in caption.split() if word.lower() in vocab.stoi]
    print(words)
    
    return torch.stack([torch.LongTensor([vocab.stoi[word]]) for word in words]).squeeze()