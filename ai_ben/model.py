import math
import torch
import numpy as np
import torch.nn as nn
from train import predict
import torch.nn.functional as F
from process import pre_process

"""
Transformer model
"""
class TransformerModel(nn.Module):
    """
    Input: ntoken - integer containing the amount of total tokens
           ninp - integer containing number of input layers
           nhead - integer containing the number of heads in the multiheadattention models
           nhid - integer containing the dimension of the feedforward network model in nn.TransformerEncoder
           nlayers - integer containing the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    Description: Initailize transormer model class creating the appropiate layers
    Output: None
    """
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, padding_idx=143):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout) #Positional encoding layer
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) #Encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) #Wrap all encoder nodes (multihead)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding_idx) #Initial encoding of imputs embed layers
        self.ninp = ninp #Number of input items
        self.decoder = nn.Linear(ninp, ntoken) #Decode layer
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    """
    Input: sz - integer containing the size of the matrix (square)
    Description: masks inputs and their future postions
    Output: pytorch tensor containing values used for mask (upper triangle set to 0)
    """
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) #Set false values to infinite
        return mask

    """
    Input: None
    Description: set the intial weights
    Output: None
    """
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    """
    Input: src - pytorch tensor containing the input sequence for the model
           src_mask - pytorch tensor containing the src mask to multiply the input sequence by
    Description: forward pass of the model
    Output: pytorch tensor containing soft max probability for each token of the sequence
    """
    def forward(self, src, src_mask):
        memory_mask = src_mask
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask) #Encoder memory
        output = self.decoder(output) #Linear layer
        output = self.softmax(output) #Get softmax probability
        return output

"""
Encode input vectors with posistional data
"""
class PositionalEncoding(nn.Module):
    """
    Input: d_model - integer containing the size of the data model input
    Description: Initailize positional encoding layer
    Output: None
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    """
    Input: x - pytorch tensor containing the input data for the model
    Description: forward pass of the positional encoding layer
    Output: pytorch tensor containing positional encoded data (floats)
    """
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
