"""
the attention here is scaled multihead attention.

this is to help understand how the key and value are got for a transformer layer,
either a encoder layer or a decoder layer.

code is from transformers/modeling_bert.py with slightly modification.

We here focus on analyzing how to get the query, key, and value.
"""

import torch
# from torch import nn 


def q_proj(hidden_states):
    """
    use a matrix to transform the hidden states to query

    nn.Linear(hidden_size_1, hidden_size_2)
    """
    pass

def k_proj(hidden_states):
    pass

def v_proj(hidden_states):
    pass

def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
    """
    add a head dim, and make sure the last 2 dims are [length, hidden_size] finally.
    x.shape = (batch, length, hidden_size)
    new_x_shape = (batch, length, heads, head_size)
    """
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)  # split the last dim
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)  # changed shape to (batch, heads, length, head_size)

def get_kvs(
    hidden_states,
    attention_mask = None,
    head_mask = None,
    encoder_hidden_states = None,
    encoder_attention_mask = None,
    past_key_value = None,
    is_decoder=False
):
    mixed_query_layer = q_proj(hidden_states)

    # cross attention is only in decoder and with encoder_hidden_states already caculated.
    is_cross_attention = encoder_hidden_states is not None 

    """
    there are 4 conditions, the first 2 conditions are related to cross attention, 
    and the last 2 conditions are realted to self attention, for both encoder and (decoder in training mode).

    condition 1: corss attention with past_kv, for 

                 cross attention during inference

    condition 2: cross attention w/o past_kv, for 

                 training_mode cross attention 
                 1st step inference cross attention.

    condition 3: decoder self attention with past_kv, pay attention that inputting only 1 token during inference, for 

                 self attention during inference 
                 
    condition 4: encoder or decoder self attention, for 

                 encoder self attention, 
                 decoder self attention in training mode, 
                 decoder self attention for 1st step during inference.
    """
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:  # get key and value from encoder_hidden_states 
        key_layer = transpose_for_scores(k_proj(encoder_hidden_states))
        value_layer = transpose_for_scores(v_proj(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = transpose_for_scores(k_proj(hidden_states))
        value_layer = transpose_for_scores(v_proj(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = transpose_for_scores(k_proj(hidden_states))
        value_layer = transpose_for_scores(v_proj(hidden_states))

    query_layer = transpose_for_scores(mixed_query_layer)

    if is_decoder:
        past_key_value = (key_layer, value_layer)
    
    outputs = (query_layer, key_layer, value_layer)

    if is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs