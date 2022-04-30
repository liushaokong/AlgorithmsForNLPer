"""
To show how to convert lengths and mask of sentences.
"""

import torch

def generate_mask_from_lengths(sentence_lengths,
                               max_sequence_length=None,
                               batch_size=None
                               ):
    """
    generate sentence masks from sentence lengths.
    return masks with shape = (batch_size, max_sequence_length)
    """

    if not isinstance(sentence_lengths, list):
        sentence_lengths = [length.item() for length in sentence_lengths]
    
    if batch_size == None:
        batch_size = len(sentence_lengths)
    else:
        assert batch_size == len(sentence_lengths), "input batch size not match the inferenced."
    
    if max_sequence_length == None:
        max_sequence_length = max(sentence_lengths)
    else:
        pass  # add your own processing here.
    
    # init a mask
    mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.float)

    for id, length in enumerate(sentence_lengths):
        mask[id, :length] = 1  # set the unpadded tokens positions to 1
    
    return mask


def generate_lengths_by_mask(mask, return_tensor=True):
    """
    mask: mask of a batch with dtype=torch.tensor
    return: a list of lengths, or a tensor
    """
    lengths = torch.sum(mask, axis=1).to(int)  # add values by row

    if not return_tensor:
        lengths = [length.item() for length in lengths]
    
    return lengths
