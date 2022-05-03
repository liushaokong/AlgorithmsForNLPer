"""
a tensorflow transfer learning example of using pretrained embedding.
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf


def save_pretrained_embedding(ckpt_path, emb_path, emb_name):
    """
    save the embedding tensor of a pretrained tf checkpoint as embedding.np

    ckpt_path: a pretrained tf ckpt model
    emb_path: path to save the extraced embedding
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)

    # check name if necessary
    # trainable_variables = tf.trainable_variables()
    # for variable in trainable_variables:
    #     print(variable.name())

    embedding = reader.get_tensor(emb_name)
    np.save(file=emb_path, arr=embedding)


def load_pretrained_embedding(emb_path):
    """
    load pretrained embedding.np
    """
    embedding = np.load(emb_path)
    return embedding 


def get_encoder_input(features, mode, params, source_embedding=None):
    """
    sample code modified based on 
    THUNLP-MT/THUMT/blob/tensorflow/thumt/models/transformer.py/encoding_graph

    initializer is optional based on whether source_embedding is used.
    """
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)

    # use source embedding to init the param
    if source_embedding is not None:
        initializer = tf.constant_initializer(source_embedding)
        trainable = False  # fixed source embedding
    else:
        initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)
        trainable = True

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer,
                                        trainable=trainable  # add trainable here, default=True
                                        )

    # bias is not taken into account, which should be considered in practice.
    bias = tf.get_variable("bias", [hidden_size])  

    inputs = tf.gather(src_embedding, src_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)

    inputs = inputs * tf.expand_dims(src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)

    return encoder_input