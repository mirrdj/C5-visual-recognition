import torch
import numpy as np
import torch.nn as nn
from torchtext.vocab import FastText
from transformers import BertTokenizer, BertModel
from utils import adjust_length_last_dim

class IncreaseDim(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(IncreaseDim, self).__init__()
        #Increase the dimension of the FastText embeddings to match the ones from the images
        self.linear = nn.Linear(input_size, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(x)
        x = self.linear(x)
        # x = x / x.pow(2).sum(dim=1, keepdim=True).sqrt()
        return x


class TorchTextNet(nn.Module):
    def __init__(self, embedding_dim):
        super(TorchTextNet, self).__init__()
        self.vocab = FastText('en')
        self.model = nn.Embedding.from_pretrained(self.vocab.vectors, freeze=False) 
        self.increase = lambda d: adjust_length_last_dim(d, embedding_dim) # Padding
        # self.increase = IncreaseDim(300, embedding_dim) # Linear layer
    
    def get_vocab(self):
        return self.vocab

    def forward(self, index_tensor_list):
        embedd_list = []
        for index_tensor in index_tensor_list:
            batched_index_tensor = index_tensor.unsqueeze(0)
            word_embeddings = self.model(batched_index_tensor)
            
            # Increase dim
            word_embeddings = self.increase(word_embeddings)

            # Perform aggregation
            mean_vector = torch.mean(word_embeddings, dim=1)
            embedd_list.append(mean_vector)
        
        final_embeddings = torch.cat(embedd_list, dim=0)

        return final_embeddings

# https://huggingface.co/docs/transformers/en/model_doc/bert
# Implementation from https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
class TorchTextNetBert(nn.Module):
    def __init__(self, embedding_dim, device, model_name='bert-base-uncased'):
        super(TorchTextNetBert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.increase = IncreaseDim(768, embedding_dim) #BERT output is 768
        self.device = device

    #We don't need to check which words are in the vocabulary, as BERT assigns authomatically a [UNK] token to them
    def get_vocab(self):
        return self.tokenizer.vocab.keys()

    
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

        word_embeddings = self.increase(word_embeddings)

        return word_embeddings