import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from ..Neural_Networks import *
from ..block_setting import *



R_Q = 0.1
R_D = 0.1



def snake(
        tensor,
        mask = None,
        trainable = True,
        scope = 'snake'
        ):
    with tf.variable_scope(scope):
        alpha = bias(
                shape = shape(tensor)[-1],
                dtype = tensor.dtype,
                trainable = trainable,
                scope = 'alpha'
                )
        alpha = tf.reshape(
                alpha,
                [1] * (len(shape(tensor)) - 1) + shape(tensor)[-1:]
                )
        tensor += tf.square(tf.sin(alpha * tensor)) / tf.maximum(alpha, 1e-8)
        if exists(mask): tensor *= mask

    return tensor



def res_blk(
        tensor,
        features,
        kernels_list = [3, 1],
        dilations_list = [1, 1],
        compress = 2,
        weight_func = None,
        bias_func = None,
        mask = None,
        trainable = True,
        scope = 'res_blk',
        ):
    with tf.variable_scope(scope):
        assert len(kernels_list) == len(dilations_list)
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            for n, (kernels, dilations) in enumerate(zip(kernels_list, dilations_list)):
                res_out = snake(
                        tensor = res_out,
                        mask = res_mask,
                        trainable = trainable,
                        scope = f'snake_{n:02d}'
                        )
                res_out, res_mask = convolution(
                    tensor = res_out,
                    rank = 1,
                    filters = features // (1 if n == len(kernels_list) - 1 else compress),
                    kernels = kernels,
                    dilations = dilations,
                    mask = res_mask,
                    quantization = R_Q,
                    quantization_blocks = kernels,
                    weight_function = weight_func,
                    bias_function = bias_func,
                    trainable = trainable,
                    scope = f'conv1d_{n:02d}'
                    )

        sc_out, sc_mask = shortcut(
                tensor = sc_out,
                features = features,
                weight_function = weight_func,
                bias_function = bias_func,
                mask = sc_mask,
                blocks = shape(sc_out)[-1],
                trainable = trainable,
                scope = 'skip'
                )

        tensor = (lambda a, b: a+ b)(
                *map(lambda a, b: a * b,
                    (res_out, sc_out),
                    res_scale(0.9, tensor.dtype, trainable = trainable, scope = 'coeff'))
                )

        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask



def network(
        tensor,
        condition = None,
        filters = 32,
        num_layers_list = [1, 1, 1, 1],
        ratios_list = [8, 5, 4, 2],
        kernels_list = [7, 3, 7],
        dilation_base = 2,
        compress = 2,
        weight_func = None,
        bias_func = None,
        mask = None,
        training = True,
        scope = 'bit_dec',
        reuse = tf.AUTO_REUSE,
        ):
    with tf.variable_scope(scope, reuse):
        assert len(num_layers_list) == len(ratios_list)

        multiplier = int(2 ** len(ratios_list))

        decoded, mask_dec = convolution(
                tensor = tensor,
                rank = 1,
                filters = multiplier * filters,
                kernels = kernels_list[0],
                mask = mask
                trainable = training,
                scope = 'conv1d_dec'
                )

        if exists(condition):
            cond_proj, _ = convolution(
                    tensor = condition[Ellipsis, None],
                    rank = 1,
                    filters = multiplier * filters,
                    kernels = kernels_list[0],
                    trainable = training,
                    scope = 'conv1d_cond_0'
                    )
            cond_proj = tf.nn.silu(cond_proj)
            cond_proj, _ = convolution(
                    tensor = cond_proj,
                    rank = 1,
                    filters = multiplier * filters,
                    kernels = kernels_list[0],
                    trainable = training,
                    scope = 'conv1d_cond_1'
                    )
            cond_proj = tf.nn.silu(cond_proj)

            decoded += cond_proj
            if exists(mask): decoded *= mask_dec

        decoded = decoded[Ellipsis, None]

        for idx, (ratio, num_layers) in enuemrate(zip(reversed(ratios_list), num_layers_list)):
            multiplier //= 2

            with tf.variable_scope(f'{idx:02d}_multi_{multiplier:03d}'):
                decoded = snake(
                        tensor = decoded,
                        mask = mask_dec,
                        trainable = training,
                        scope = 'snake'
                        )
                if ratio > 1:
                    decoded, mask_dec = deconvolution(
                            tensor = decoded,
                            rank = 1,
                            filters = multiplier * filters,
                            kernels = ratio * 2,
                            strides = ratio,
                            mask = mask_dec,
                            weight_function = weight_func,
                            bias_function = bias_func,
                            trainable = training,
                            scope = 'conv1d_up'
                            )
                else:
                    decoded, mask_dec = convolution(
                            tensor = decoded,
                            rank = 1,
                            filters = multiplier * filters,
                            kernels = 3,
                            strides = 1,
                            mask = mask_dec,
                            weight_function = weight_func,
                            bias_function = bias_func,
                            trainable = training,
                            scope = 'conv1d'
                            )

                for n_layer in range(num_layers):
                    decoded, mask_dec = res_blk(
                            tensor = decoded,
                            features = multiplier * filters,
                            kernels_list = [kernels_list[1], 1],
                            dilations_list = [dilation_base ** n_layer, 1],
                            compress = compress,
                            weight_func = weight_func,
                            bias_func = bias_func,
                            mask = mask_dec,
                            trainable = training,
                            scope = f'blk_{n_layer:02d}',
                            )

        decoded = snake(
                tensor = decoded,
                mask = mask_dec,
                trainable = training,
                scope = 'snake'
                )
        decoded, mask_dec = convolution(
                tensor = decoded,
                kernels = kernels_list[2],
                weight_function = weight_func,
                bias_function = bias_func,
                trainable = training,
                scope = 'conv1d_out')
        decoded = tf.tanh(decoded)
        if exists(mask): decoded *= mask_dec

    return decoded, mask_dec



