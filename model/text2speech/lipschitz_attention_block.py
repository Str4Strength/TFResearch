import tensorflow._api.compat.v1 as tf
tf.disable_v2_behavior()

from functools import partial

from ..Neural_Network import *
from .block_setting import *
from .cond_weight import *


# core tensor : repeated text-oriented tokens, matched into speech
# condition tensor : speaker's style, for speaking tendency
# cross condition tensor : original text, for context understanding



def lipschitz_attention_block(
        tensor,
        companion = None,
        condition = None,
        features = 512,
        kernels = 5,
        iteration = 5,
        heads = 8,
        groups = 16,
        causal = False,
        activation = mish,
        mask = None,
        saved_state = None,
        quantization = 0.0,
        weight_function = spectral_normalize,
        trainable = True,
        scope = 'lips_attn_blk',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse):
        tensor, mask = linear(
                tensor = tensor,
                out_features = features,
                mask = mask,
                quantization = quantization,
                quantization_blocks = shape(tensor)[-1],
                weight_function = weight_function,
                trainable = trainable,
                scope = 'linear'
                )

        L = shape(tensor)[1]
        # exponential scale 고려하여 가려줄 부분만 masking값 부여
        causal_mask = -1e9 * tf.cast(tf.range(L)[:, None] < tf.range(L)[None], dtype = tensor.dtype)[None, Ellipsis, None]

        # abs pos enc skipped since rotary pos enc internally applied in attention
        for n in range(iteration):

            with tf.variable_scope(f'{n:02d}'):

                with tf.variable_scope('major_block'):
                    residual, _, _ = attention(
                            query = tensor,
                            key = companion if exists(companion) else tensor,
                            value = companion if exists(companion) else tensor,
                            out_features = features,
                            heads = heads,
                            core_operation = l2_attention,
                            mask_query = mask,
                            mask_key = mask,
                            mask_value = mask,
                            saved_state = getattr(saved_state, 'attn', None),
                            attention_bias = causal_mask if causal else None,     # CAUTION: causal
                            quantization = quantization,
                            quantization_blocks = features,
                            query_weight_function = weight_function,
                            key_weight_function = weight_function,
                            value_weight_function = weight_function,
                            trainable = trainable,
                            scope = 'attn'
                            )
                    #shortcut, mask = shortcut(
                    #        tensor = tensor,
                    #        features = features,
                    #        mask = mask,
                    #        quantization = quantization,
                    #        quantization_blocks = features,
                    #        weight_function = weight_function,
                    #        trainable = trainable,
                    #        scope = 'scut'
                    #        )
                    #scale = weight_(
                    #        shape = [],
                    #        dtype = tensor.dtype,
                    #        init = tf.zeros_initializer,
                    #        function = lambda x: tf.minimum(tf.math.abs(x), 1),
                    #        trainable = trainable,
                    #        scope = 'resc'
                    #        )
                    #tensor = (scale * residual) + (tf.sqrt(1 - (scale ** 2)) * shortcut)

                with tf.variable_scope('minor_block'):
                    if causal:
                        xpnd, xpnd_m = getattr(saved_state, 'xpnd', None), getattr(saved_state, 'xpnd_mask', None)
                        if exists(xpnd):
                            r, r_m = link_memory(tensor, mask, tensor_saved = xpnd, mask_saved = xpnd_m)
                            setattr(saved_state, 'xpnd', r[:, - kernels + 1:])
                            if exists(r_m): setattr(saved_state, 'xpnd_mask', r_m[:, - kernels + 1:])
                        else:
                            r = tf.pad(tensor, ((0, 0), (kernels - 1, 0), (0, 0)))
                            if exists(mask): r_m = tf.pad(mask, ((0, 0), (kernels - 1, 0), (0, 0)))
                    xpnd_weight = conditional_weight(
                            condition = condition,
                            kernels = kernels,
                            in_features = features,
                            out_features = 2 * features,
                            groups = groups,
                            quantization = quantization,
                            weight_function = weight_function,
                            trainable = True,
                            scope = 'xpnd_weight'
                            ) if exists(condition) else None
                    residual, res_mask = convolution(
                            tensor = r,
                            rank = 1,
                            filters = 2 * features,
                            kernels = kernels,
                            padding = 'VALID' if causal else 'SAME',
                            groups = groups,
                            mask = r_m,
                            weight = xpnd_weight,
                            quantization = quantization,
                            quantization_blocks = np.prod(kernels),
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'xpnd'
                            )

                    residual = activation(residual)

                    if causal:
                        sqze, sqze_m = getattr(saved_state, 'sqze', None), getattr(saved_state, 'sqze_mask', None)
                        if exists(sqze):
                            r, r_m = link_memory(residual, res_mask, tensor_saved = sqze, mask_saved = sqze_m)
                            setattr(saved_state, 'sqze', r[:, - kernels + 1:])
                            if exists(r_m): setattr(saved_state, 'sqze_mask', r_m[:, - kernels + 1:])
                        else:
                            r = tf.pad(residual, ((0, 0), (kernels - 1, 0), (0, 0)))
                            if exists(mask): r_m = tf.pad(mask, ((0, 0), (kernels - 1, 0), (0, 0)))

                    sqze_weight = conditional_weight(
                            condition = condition,
                            kernels = kernels,
                            in_features = 2 * features,
                            out_features = features,
                            groups = groups,
                            quantization = quantization,
                            weight_function = weight_function,
                            trainable = True,
                            scope = 'sqze_weight'
                            ) if exists(condition) else None
                    residual, _ = convolution(
                            tensor = r,
                            rank = 1,
                            filters = features,
                            kernels = kernels,
                            padding = 'VALID' if causal else 'SAME',
                            groups = groups,
                            mask = r_m,
                            weight = sqze_weight,
                            quantization = quantization,
                            quantization_blocks = np.prod(kernels),
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'sqze'
                            )

                    scale = weight_(
                            shape = [],
                            dtype = tensor.dtype,
                            init = tf.zeros_initializer,
                            function = lambda x: tf.minimum(tf.math.abs(x), 1),
                            trainable = trainable,
                            scope = 'resc'
                            )
                    tensor = (scale * residual) + (tf.sqrt(1 - (scale ** 2)) * tensor)

            return tensor, mask



