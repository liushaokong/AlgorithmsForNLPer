"""
this is a file showing how the 3 kinds of position embedding work.
the code is extrqacted from thumt and transfomers/models/modeling_bert.py
the 3 kinds of embedding are: {sin_cos, learnable, relative}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from turtle import forward
import torch
from torch import nn


class TrigonometricPositionalEmbedding(torch.nn.Module):
    """
    from THUMT-torch
    """
    def __init__(self):
        super(TrigonometricPositionalEmbedding, self).__init__()

    def forward(self, inputs):
        if inputs.dim() != 3:  # (batch, length, channels)
            raise ValueError("The rank of input must be 3.")

        length = inputs.shape[1]
        channels = inputs.shape[2]
        half_dim = channels // 2  # 256 => 128

        positions = torch.arange(length, dtype=inputs.dtype,
                                 device=inputs.device)  # tensor([0, 1, 2, 3, ...])
        
        # different embedding functions for different feature dimensions
        dimensions = torch.arange(half_dim, dtype=inputs.dtype,
                                  device=inputs.device)  # e.g. tensor([0, 1, 2, ..., 127])

        # pesudo code for postion embedding in the paper

        # k-th dimention of a feature
        # if k = 2 * i: pe = sin(pos / (10000 ^ (k/d)))
        # if k = 2 * i + 1: pe = cos(pos / (10000 ^ (k/d))
        
        # slightly different here, each position includes sin and cos embedding information.
        scale = math.log(10000.0) / float(half_dim - 1)
        dimensions.mul_(-scale).exp_()  # torch style inplace

        scaled_time = positions.unsqueeze(1) * dimensions.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)

        if channels % 2 == 1:  # pad 0
            pad = torch.zeros([signal.shape[0], 1], dtype=inputs.dtype,
                              device=inputs.device)
            signal = torch.cat([signal, pad], axis=1)

        return inputs + torch.reshape(signal, [1, -1, channels]).to(inputs)


class LearnablePositionEmbedding(torch.nn.Module):
    """
    define a postion embedding matrix, use postion id to lookup.
    """
    def __init__(self, max_position_embeddings, hidden_size):
        self.max_postion_embeddings = max_position_embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, embeddings):
        # embeddings = inputs_embeds + token_type_embeddings
        position_ids = torch.arange(self.max_position_embeddings).expand((1, -1))
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        return embeddings

class RelativePositionEmbedding(torch.nn.Module):
    def __init__(self, 
                 max_position_embeddings, 
                 hidden_size,
                 position_embedding_type="relative_key"):
        # position_embedding_type = {relative_key, relative_key_query}
        self.position_embedding_type = position_embedding_type  
        self.hidden_size = hidden_size
        # max_position_embeddings: n
        self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, hidden_size)  # (2n-1, hidden) e.g. n=3

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # query_layer.shape = (batch, length, channels)

        score = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = query_layer.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=query_layer.device).view(-1, 1)  # tensor([[0], [1], [2], [3], [4]])
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=query_layer.device).view(1, -1)  # tensor([0, 1, 2, 3, 4])

            distance = position_ids_l - position_ids_r 
            # tensor([[ 0, -1, -2, -3],
            #         [ 1,  0, -1, -2],
            #         [ 2,  1,  0, -1],
            #         [ 3,  2,  1,  0]])
   

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)  # change range from (-n, n) to (0, 2n-1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            # scores_key + scores_query (optional)
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)  # einstein sum
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        return context_layer
