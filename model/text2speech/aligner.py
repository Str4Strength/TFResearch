import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *

from .style_encoder import *
from .retrieval import *
from .attention_block import *
from .convolution_block import *



def aligner(
        text,
        image,
        speaker,
        features = 512,
        text_capacity = 10000,
        text_features = 512,
        condition_features = None,
        condition_compress = 4,
        encodings = 1,
        decodings = 1,
        shift = 1,
        text_mask = None,
        image_mask = None,
        trainable = True,
        scope = 'algn'
        ):
    with tf.variable_scope(scope):
        core_operation = general_attention
        head = max(features // 128, 1)
        window = 5
        kernel = 3
        weight_groups = 1
        normalization_groups = 32

        # style
        condition = style_encoder(
                tensor = image,
                features = condition_features,
                iteration = condition_compress,
                mask = image_mask,
                trainable = trainable,
                scope = 'cond'
                )

        # token embedding
        array = retrieval(
                query = text,
                capacity = text_capacity,
                size = text_features,
                dtype = image.dtype,
                zero_pad = True,
                scale = True,
                trainable = trainable,
                scope = 'tknz'
                )

        # encoding block
        array, array_mask, _, _ = attention_block(
                tensor = array,
                condition = condition,
                iteration = encodings,
                features = text_features,
                core_operation = general_attention,
                head = head,
                kernel = kernel,
                window = window,
                weight_groups = weight_groups,
                normalization_groups = normalization_groups,
                causality = True,
                mask = tf.tile(text_mask[Ellipsis, None], [1, 1, text_features]) if exists(text_mask) else None,
                trainable = trainable,
                scope = 'encd'
                )

        # image projection
        tensor, _ = convolution(
                tensor = image,
                rank = 1,
                filters = features,
                kernel = 1,
                trainable = trainable,
                scope = 'frwd/proj'
                )
        if exists(image_mask):
            tensor *= image_mask[Ellipsis, :1]
            tensor_mask = reconstruct_mask(features, image_mask)

        # decoding block
        tensor, tensor_mask, _, _ = attention_block(
                tensor = tensor,
                condition = condition,
                iteration = decodings,
                features = features,
                core_operation = core_operation,
                head = head,
                kernel = kernel,
                weight_groups = weight_groups,
                normalization_groups = normalization_groups,
                causality = True,
                mask = tensor_mask,
                trainable = trainable,
                scope = 'frwd'
                )

        with tf.variable_scope('mtch'):

            with tf.variable_scope('fnrm'):
                gamma, beta = None, None
                tensor = normalization(
                        tensor = tensor,
                        groups = normalization_groups,
                        gamma = gamma,
                        beta = beta,
                        mask = tensor_mask,
                        trainable = trainable,
                        scope = 'norm'
                        )

            tensor = gelu(tensor)
            if exists(tensor_mask): tensor *= tensor_mask

            tensor, tensor_mask, attention_map = attention(
                    query = tensor,
                    key = array,
                    value = array,
                    out_features = features,
                    core_operation = general_attention,
                    head = 1,
                    mask_query = tensor_mask,
                    mask_key = array_mask,
                    mask_value = array_mask,
                    trainable = trainable,
                    scope = 'attn'
                    )
            align_map = attention_map[Ellipsis, 0]

            with tf.variable_scope('bnrm'):
                gamma, beta = None, None
                tensor = normalization(
                        tensor = tensor,
                        groups = normalization_groups,
                        gamma = gamma,
                        beta = beta,
                        mask = tensor_mask,
                        trainable = trainable,
                        scope = 'norm'
                        )

            tensor = gelu(tensor)
            if exists(tensor_mask): tensor *= tensor_mask

        tensor, tensor_mask, _, _ = attention_block(
                tensor = tensor,
                condition = condition,
                features = features,
                iteration = decodings,
                core_operation = core_operation,
                head = head,
                kernel = kernel,
                weight_groups = weight_groups,
                normalization_groups = normalization_groups,
                causality = True,
                mask = tensor_mask,
                trainable = trainable,
                scope = 'bkwd'
                )

        tensor, _ = convolution(
                tensor = tensor,
                rank = 1,
                filters = shape(image)[-1] * shift,
                kernel = 1,
                trainable = trainable,
                scope = 'bkwd/proj'
                )

        return tensor, align_map


