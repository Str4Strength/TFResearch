import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *

from .style_encoder import *
from .retrieval import *
from .attention_block import *



def duration_allocator(
        text,
        speaker,
        features = 512,
        condition_features = 512,
        text_capacity = 10000,
        speaker_capacity = 100000,
        iteration = 1,
        dtype = tf.float32,
        text_mask = None,
        quantization = 0.0,
        trainable = True,
        scope = 'droc'
        ):
    with tf.variable_scope(scope):
        head = max(features // 128, 1)
        kernel = 3
        weight_groups = 4
        normalization_groups = 32

        condition = retrieval(
                query = tf.reshape(speaker, (shape(speaker)[0], 1)),
                capacity = speaker_capacity,
                size = condition_features,
                dtype = dtype,
                zero_pad = True,
                scale = True,
                trainable = trainable,
                scope = 'cond'
                )

        sequence = retrieval(
                query = text,
                capacity = text_capacity,
                size = features,
                dtype = dtype,
                zero_pad = True,
                scale = True,
                trainable = trainable,
                scope = 'tknz'
                )

        sequence, _, _, _ = attention_block(
                tensor = sequence,
                condition = condition,
                iteration = iteration,
                features = features,
                core_operation = general_attention,
                head = head,
                kernel = kernel,
                weight_groups = weight_groups,
                normalization_groups = normalization_groups,
                causality = trainable,
                mask = tf.tile(text_mask[Ellipsis, None], [1, 1, features]) if exists(text_mask) else None,
                quantization = quantization,
                trainable = trainable,
                scope = 'allc'
                )

        duration, _ = convolution(
                tensor = sequence,
                rank = 1,
                filters = 1,
                kernel = 1,
                trainable = trainable,
                scope = 'conv'
                )
        duration = duration[Ellipsis, 0]
        if exists(text_mask): duration *= text_mask

        return duration

