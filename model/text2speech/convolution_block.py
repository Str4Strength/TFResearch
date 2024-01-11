import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *
from .block_setting import *



def convolution_block(
        tensor,
        condition = None,
        iteration = 5,
        features = 512,
        kernel = 5,
        weight_groups = 1,
        normalization_groups = 16,
        causality = False,
        activation = mish,
        mask = None,
        quantization = 0.0,
        weight_function = None,
        trainable = True,
        scope = 'conv_blk',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse = reuse):

        skip, skip_mask = shortcut(
                tensor = tensor,
                features = features,
                mask = mask,
                quantization = quantization,
                quantization_blocks = shape(tensor)[-1],
                weight_function = weight_function,
                trainable = trainable,
                scope = 'scut'
                )
        res_mask = reconstruct_mask(features // 4, mask)

        for n in range(iteration):

            with tf.variable_scope(f'{n:02d}'):

                with tf.variable_scope('bneck_block'):
                    residual, _ = convolution(
                            tensor = tensor,
                            rank = 1,
                            filters = features // 4,
                            kernel = 1,
                            groups = weight_groups,
                            quantization = quantization,
                            quantization_blocks = shape(tensor)[-1],
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'conv'
                            )
                    if exists(mask): residual *= res_mask

                    if exists(condition):
                        gamma_beta, _ = convolution(
                                tensor = tf.reshape(condition, [shape(condition)[0], 1, shape(condition)[-1]]),
                                rank = 1,
                                filters = 2 * normalization_groups,
                                kernel = 1,
                                quantization = quantization,
                                quantization_blocks = features // 4,
                                weight_function = weight_function,
                                trainable = trainable,
                                scope = 'gmbt'
                                )
                        gamma, beta = gamma_beta[Ellipsis, :normalization_groups, None], gamma_beta[Ellipsis, normalization_groups:, None]
                    else:
                        gamma, beta = None, None

                    residual = normalization(
                            tensor = residual,
                            groups = normalization_groups,
                            gamma = gamma,
                            beta = beta,
                            mask = res_mask,
                            gamma_function = weight_function,
                            trainable = trainable,
                            scope = 'norm'
                            )
                    residual = activation(residual)
                    if exists(mask): residual *= res_mask

                with tf.variable_scope('major_block'):

                    if causality: residual = tf.pad(residual, ((0, 0), (kernel - 1, 0), (0, 0)))
                    residual, _ = convolution(
                            tensor = residual,
                            rank = 1,
                            filters = features,
                            kernel = kernel,
                            padding = 'VALID' if causality else 'SAME',
                            groups = weight_groups,
                            quantization = quantization,
                            quantization_blocks = shape(tensor)[-1],
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'conv'
                            )

                    if exists(condition):
                        gamma_beta, _ = convolution(
                                tensor = tf.reshape(condition, [shape(condition)[0], 1, shape(condition)[-1]]),
                                rank = 1,
                                filters = 2 * normalization_groups,
                                kernel = 1,
                                quantization = quantization,
                                quantization_blocks = features,
                                weight_function = weight_function,
                                trainable = trainable,
                                scope = 'gmbt'
                                )
                        gamma, beta = gamma_beta[Ellipsis, :normalization_groups, None], gamma_beta[Ellipsis, normalization_groups:, None]
                    else:
                        gamma, beta = None, None

                    residual = normalization(
                            tensor = residual,
                            groups = normalization_groups,
                            gamma = gamma,
                            beta = beta,
                            mask = skip_mask,
                            gamma_function = weight_function,
                            trainable = trainable,
                            scope = 'norm'
                            )

                scale = weight_(
                        shape = [],
                        dtype = tensor.dtype,
                        init = tf.zeros_initializer,
                        trainable = trainable,
                        scope = 'resc'
                        )
                scale = tf.minimum(tf.math.abs(scale), 1)
                tensor = scale * skip + residual
                skip = tensor

        return tensor, skip_mask

