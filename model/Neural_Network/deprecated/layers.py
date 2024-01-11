import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import nn_ops, gen_nn_ops, random_ops

import numpy as np

import scipy.signal

import re
import math
import random
import string

from functools import partial
from termcolor import cprint

from .functions import *
from .rot_pos_enc import *
#from .upfirdn_2d import *



# base structure of tensor : batch-level, spatial-level, channels
# for all layers, mask must be in a full shape


def shape(x):
    static, dynamic = [x.shape.as_list(), tf.shape(x)]
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def weight_(
        shape,
        dtype,
        init=None,
        gain=1,
        use_wscale=False,
        lrmul=1,
        function=None,
        trainable=True,
        scope='weight',
        ):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)

    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    if init is None: init = tf.initializers.random_normal(0, init_std)
    weight = tf.get_variable(scope, shape=shape, dtype=dtype, initializer=init, trainable=trainable) * runtime_coef

    if exists(function):
        if isinstance(function, list) or isinstance(function, set):
            for f in function:
                if exists(f): weight = f(weight)
        else:
            weight = function(weight)

    return weight



def bias_(
        shape,
        dtype,
        init=None,
        lrmul=1,
        function=None,
        trainable=True,
        scope='bias',
        ):
    if init is None: init = tf.zeros_initializer
    bias = tf.get_variable(scope, shape=shape, dtype=dtype, initializer=init, trainable=trainable) * lrmul

    if function is not None:
        if isinstance(function, list) or isinstance(function, set):
            for f in function: bias = f(bias)
        else:
            bias = function(bias)

    return bias


def reconstruct_mask(
        features,
        mask=None,
        axis=-1
        ):
    if not exists(mask): return

    full, is_neg = len(shape(mask)), int(axis < 0)
    front, back = axis, - axis - 1

    mask = tf.reduce_max(mask, axis=axis, keepdims=True)
    mask = tf.tile(mask, [1] * (front + is_neg * full)  + [features] + [1] * (back + (1 - is_neg) * full))
    return mask


def quantization_noise(
        weight,
        in_features,
        out_features,
        p,
        block_size,
        ):
    if p <= 0: return weight

    #in_features = (in_features,) if isinstance(in_features, int) else tuple(in_features)
    #out_features = (out_features,) if isinstance(out_features, int) else tuple(out_features)
    if not isinstance(in_features, tuple):
        in_features = tuple(in_features) if isinstance(in_features, list) else (in_features,)
    if not isinstance(out_features, tuple):
        out_features = tuple(out_features) if isinstance(out_features, list) else (out_features,)

    weight_shape, features = shape(weight), (*in_features, *out_features)
    is_conv = tuple(weight_shape) != features
    if is_conv: kernel_shape = weight_shape[:-len(features)]

    if not is_conv:
        assert np.prod(in_features) % block_size == 0
    else:
        k = np.prod(kernel_shape)
        if k == 1:
            assert np.prod(in_features) % block_size == 0
        else:
            assert k % block_size == 0

    if is_conv and k != 1:
        mask = tf.greater(tf.random.uniform(shape=features, maxval=1, dtype=weight.dtype), p)
    else:
        mask = tf.greater(tf.random.uniform(shape=[np.prod(in_features) // block_size, *out_features], maxval=1, dtype=weight.dtype), p)
        mask = tf.reshape(tf.tile(mask[:, None], [1, block_size, * (1,) * len(out_features)]), features)

    if is_conv:
        mask = tf.tile(mask[(None,) * (len(weight_shape) - len(features))], (* kernel_shape, * (1,) * len(features)))

    weight = tf.where(mask, weight, tf.zeros_like(weight)) / (1 - p)

    return weight



"""
def linear(
        tensor,
        in_features=None,
        out_features=None,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        trainable=True,
        scope='linear'
        ):
    # mask must be in a full-shape except channels
    with tf.variable_scope(scope):
        tensor_shape, dtype = shape(tensor), tensor.dtype
        if in_features is None: in_features = tensor_shape[-1]
        if out_features is None: out_features = in_features

        in_features = [in_features] if isinstance(in_features, int) else list(in_features)
        out_features = [out_features] if isinstance(out_features, int) else list(out_features)
        ts_len, fin_len, fout_len = len(tensor_shape), len(in_features), len(out_features)
        assert fin_len <= len(string.ascii_lowercase)
        assert fout_len <= len(string.ascii_uppercase)

        in_context = string.ascii_lowercase[:fin_len]
        out_context = string.ascii_uppercase[:fout_len]
        context = '...' + in_context + ',' + in_context + out_context + '->...' + out_context

        weight_shape = [np.prod(in_features), np.prod(out_features)]
        bias_shape = [1] * (ts_len - fin_len) + out_features

        weight = weight_(weight_shape, dtype, lrmul=lrmul, function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, quantization, quantization_blocks)

        tensor = tf.einsum(context, tensor, tf.reshape(weight, [*in_features, *out_features]))

        if bias: tensor += bias_(bias_shape, dtype, lrmul=lrmul, function=bias_function, trainable=trainable)

        if exists(mask):
            mask = tf.tile(tf.reshape(tf.reduce_max(mask, axis=range(ts_len - fin_len, ts_len)),
                tensor_shape[:-fin_len] + [1] * fout_len), bias_shape)
            tensor *= mask

        return tensor, mask
"""



def linear(
        tensor,
        in_features=None,
        out_features=None,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        trainable=True,
        scope='linear'
        ):
    # mask must be in a full-shape except channels
    with tf.variable_scope(scope):
        dtype, tensor_shape = tensor.dtype, shape(tensor)

        if in_features is None: in_features = (tensor_shape[-1], )
        in_features = (in_features,) if isinstance(in_features, int) else tuple(in_features)
        if out_features is None: out_features = in_features
        out_features = (out_features,) if isinstance(out_features, int) else tuple(out_features)

        dims, ins, outs = len(tensor_shape), len(in_features), len(out_features)

        weight = weight_((*in_features, *out_features), dtype, lrmul=lrmul, function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, in_features, out_features, quantization, quantization_blocks)

        axes = (list(range(dims - ins, dims)), list(range(ins)))
        tensor = tf.tensordot(tensor, weight, axes)

        if bias: tensor += bias_(out_features, dtype, lrmul=lrmul, function=bias_function, trainable=trainable)[(None,) * (dims - ins)]

        if exists(mask):
            mask = tf.reduce_max(mask, axis=tuple(range(dims - ins, dims)))
            mask = tf.tile(mask[(Ellipsis, * (None,) * outs)], (* (1,) * (dims - ins), *out_features))
            tensor *= mask

        return tensor, mask


'''
def normalization(
        tensor,
        target_positions=[0],
        groups=1,
        scale=True,
        shift=True,
        mask=None,
        condition=None,
        condition_mask=None,
        epsilon=1e-8,
        momentum=.99,
        lrmul=1.0,
        gamma_function=None,
        beta_function=None,
        trainable=True,
        scope='normalize'
        ):
    """
    default tensor setting as [Batch, Channels] + Spatial_Dimensions
    0 for batch axis, of course batch-across norm required for multi-gpu
    -1 for channels axis
    1, 2, ... for specific Spatial Dimensions

    mask required in the full shape equal to the tensor
    """
    with tf.variable_scope(scope):
        if groups > 1: tensor, mask = group(tensor, groups, axis=-1, mask=mask)

        dtype = tensor.dtype
        static, dynamic = [tensor.shape.as_list(), tf.shape(tensor)]
        tensor_shape = [dynamic[i] if s is None else s for i, s in enumerate(static)]
        gamma_shape = [1 if i in ([0] + target_positions) or s is None else s for i, s in enumerate(static)]

        def get_mean_var(x, mask=None):
            if mask is None:
                mean = tf.reduce_mean(x, target_positions, keepdims=True)
                variance = tf.reduce_mean(tf.square(x), target_positions, keepdims=True) - tf.square(mean)
            else:
                mask_sum = tf.reduce_sum(mask, target_positions, keepdims=True)
                div = tf.where(tf.equal(mask_sum, 0.0), tf.zeros_like(mask_sum), 1.0/mask_sum)

                mean = tf.reduce_sum(x, target_positions, keepdims=True) * div
                variance = tf.reduce_sum(tf.square(x - mean), target_positions, keepdims=True) * div
                #mean_square = tf.reduce_sum(tf.square(x), target_positions, keepdims=True) * div
                #variance = mean_square - tf.square(mean)

            return mean, variance

        if exists(condition):
            static_, dynamic_ = [condition.shape.as_list(), tf.shape(condition)]
            condition_shape = [dynamic_[i] if s is None else s for i, s in enumerate(static_)]
            adaptive_vector = set(condition_shape[1:]) <= set(static_)

            if adaptive_vector:
                condition = tf.reshape(condition, [condition_shape[0], [-1]])
                gamma = linear(condition, out_features=tensor_shape[-1], mask=None, trainable=trainable, lrmul=lrmul,
                        weight_function=gamma_function, scope='adaptive_gamma')[0]
                gamma = tf.reshape(gamma, condition_shape[:1] + [1]*(len(tensor_shape)-2) + [-1])
                beta = linear(condition, out_features=tensor_shape[-1], mask=None, trainable=trainable, lrmul=lrmul,
                        weight_function=beta_function, scope='adaptive_beta')[0]
                beta = tf.reshape(beta, condition_shape[:1] + [1]*(len(tensor_shape)-2) + [-1])

            else:
                gamma, beta_square = get_mean_var(condition, mask=condition_mask)
                beta = tf.sqrt(beta_square)

        else:
            gamma = weight_(gamma_shape, dtype, init=tf.ones_initializer, lrmul=lrmul, function=gamma_function,
                    trainable=trainable, scope='gamma')
            beta = bias_([1]*(len(tensor_shape)-1) + tensor_shape[-1:], dtype, lrmul=lrmul, function=beta_function,
                    trainable=trainable, scope='beta')

        if 0 in target_positions:
            decay = 1 - momentum

            ema_mean = tf.get_variable('ema_mean', shape=gamma_shape, dtype=dtype, initializer=tf.zeros_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            ema_variance = tf.get_variable('ema_variance', shape=gamma_shape, dtype=dtype, initializer=tf.ones_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            smp_mean, smp_variance = get_mean_var(tensor, mask)

            mean = (decay * smp_mean) + (momentum * ema_mean)
            variance = (decay * smp_variance) + (momentum * ema_variance)

            with tf.control_dependencies([tf.assign(ema_mean, mean), tf.assign(ema_variance, variance)]):
                tensor_normed = (tensor - mean) * tf.rsqrt(tf.maximum(variance, epsilon))

        else:
            mean, variance = get_mean_var(tensor, mask)
            var_rsqrt = tf.where(tf.less_equal(tf.abs(variance), epsilon), tf.zeros_like(variance), tf.rsqrt(variance))
            tensor_normed = (tensor - mean) * var_rsqrt

        tensor = (tensor_normed * gamma) + beta

        if groups > 1: tensor, mask = ungroup(tensor, groups, axis=-1, mask=mask)

        if exists(mask): tensor *= mask

    return tensor
'''


def normalization(
        tensor,
        groups=None,
        group_size=None,
        batch=False,
        scale=True,
        shift=True,
        gamma=None,
        beta=None,
        mask=None,
        epsilon=1e-5,
        momentum=0.1,
        lrmul=1.0,
        gamma_function=None,
        beta_function=None,
        trainable=True,
        scope='normalization'
        ):
    with tf.variable_scope(scope):
        dtype, tensor_shape = tensor.dtype, shape(tensor)
        len_shape = len(tensor_shape)
        if exists(groups) and exists(group_size):
            assert tensor_shape[-1] == groups * group_size
        else:
            assert exists(groups) or exists(group_size)
            groups = tensor_shape[-1] // group_size if exists(group_size) else groups
            group_size = tensor_shape[-1] // groups if exists(groups) else group_size

        var_shape = (1,) * (len_shape - 1) + (groups, 1)

        if scale and gamma is None: gamma = weight_(var_shape, dtype, init=tf.ones_initializer, lrmul=lrmul,
                    function=gamma_function, trainable=trainable, scope='gamma')
        if shift and beta is None: beta = bias_(var_shape, dtype, lrmul=lrmul,
                function=beta_function, trainable=trainable, scope='beta')

        # ... d g
        reduce_kwargs = {'axis': (0,) * int(batch) + tuple(range(1, len_shape - 1)) + (len_shape,), 'keepdims': True}

        def _mean_(x, mask=None, **kwargs):
            if mask is None: return tf.reduce_mean(x, **kwargs)
            mask_sum = tf.reduce_sum(mask, **kwargs)
            return tf.reduce_sum(x, **kwargs) / tf.where(mask_sum > 0, mask_sum, tf.ones_like(mask_sum))

        grouped_tensor = tf.reshape(tensor, tensor_shape[:-1] + [groups, group_size])
        grouped_mask = tf.reshape(mask, tensor_shape[:-1] + [groups, group_size]) if exists(mask) else None

        smp_mean = _mean_(grouped_tensor, mask=grouped_mask, **reduce_kwargs)     # 1, ..., 1, g, 1
        smp_variance = _mean_(grouped_tensor ** 2.0, mask=grouped_mask, **reduce_kwargs) - (smp_mean ** 2.0)

        if batch:
            ema_mean = tf.get_variable('ema_mean', var_shape, dtype, initializer=tf.zeros_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            ema_variance = tf.get_variable('ema_variance', var_shape, dtype, initializer=tf.ones_initializer,
                    trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])

            if exists(mask): reduced_mask = tf.reduce_max(grouped_mask, **reduce_kwargs)[None, None]
            mean, variance = momentum * (smp_mean - ema_mean), momentum * (smp_variance - ema_variance)
            if exists(mask): mean, variacne = map(lambda t: t * reduced_mask, (mean, variance))
            mean, variance = mean + ema_mean, variance + ema_variance

            with tf.control_dependencies([tf.assign(ema_mean, mean), tf.assign(ema_variance, variance)]
                    if trainable else []):
                smp_mean, smp_variance = tf.identity(mean), tf.identity(variance)

        def _duplicate_(x, mask=None,):
            x = tf.reshape(tf.tile(x, (1,) * len_shape + (group_size,)), (* shape(x)[:-2], tensor_shape[-1]))
            if exists(mask): x *= mask
            return x

        tensor = (tensor - _duplicate_(smp_mean, mask=mask)) * tf.rsqrt(tf.maximum(_duplicate_(smp_variance), epsilon))
        tensor = tensor * _duplicate_(gamma, mask=mask) + _duplicate_(beta, mask=mask)
        if exists(mask): tensor *= mask

    return tensor



def convolution(
        tensor,
        rank,
        filters,
        kernels,
        strides=1,
        dilations=1,
        padding='SAME',
        groups=1,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        data_format=None,
        trainable=True,
        scope='convolution'
        ):
    """
    convolution of <rank>-dimensional
    """
    with tf.variable_scope(scope):
        assert len(shape(tensor)) > rank + 1

        # padding check
        padding = padding.upper()

        # rank check
        if rank not in {1, 2, 3}:
            raise ValueError('The number of spatial dimensions must be one of 1, 2 or 3 but saw {}.'.format(rank))

        # filters check
        if isinstance(filters, float): filters = int(filters)
        if exists(filters) and filters % groups != 0:
            raise ValueError('The number of filters must be evenly divisible by the number of groups.'
                             'Received: groups={}, filters={}.'.format(groups, filters))

        # channels check
        dtype = tensor.dtype
        tensor_shape = shape(tensor)
        if tensor_shape[-1] % groups != 0:
            raise ValueError('The number of input channels must be evenly divisible by the number of groups.'
                             'Received groups={}, but the input has {} channels (full input shape is {}).'.format(
                groups, tensor_shape[-1], tensor_shape))

        # kernel size control
        if isinstance(kernels, int): kernels = [kernels, ] * rank
        kernels = list(kernels)
        if len(kernels) != rank:
            raise ValueError('The `kernels` argument must be a list of {} integers.'
                             'Received: {}.'.format(rank, kernels))
        for single_size in kernels:
            assert isinstance(single_size, int)
        if not all(kernels):
            raise ValueError('`kernels` cannot contain 0(s).'
                             'Received: {}'.format(kernels))

        # internal convolution operation
        n_total_dims = len(tensor_shape)
        n_batch_dims = n_total_dims - rank - 1
        batch_dims = list(range(0, n_batch_dims))

        # mask
        if exists(mask): tensor = tensor * mask

        weight = weight_(kernels + [tensor_shape[-1]//groups, filters//groups], dtype, lrmul=lrmul,
                function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, tensor_shape[-1]//groups, filters, quantization, quantization_blocks)
        if groups > 1: weight = tf.tile(weight, [1] * rank + [1, groups])

        # manufacture shape
        tensor = tf.reshape(tensor, [-1] + tensor_shape[n_batch_dims:])
        if exists(mask): mask = tf.reshape(mask, [-1] + tensor_shape[n_batch_dims:])

        if data_format == 'channels_first': tensor = tf.transpose(tensor, [0] + list(range(2, rank + 2)) + [1])

        def reform(values, name='values'):
            if isinstance(values, int): values = [values, ] * rank
            values = list(values)

            for single_size in values: assert isinstance(single_size, int)

            if not all(values): raise ValueError('`{}` cannot contain 0(s). Received: {}'.format(name, values))

            n_value_dims = len(values)

            if n_value_dims != (rank + 2):
                if n_value_dims == 1:
                    values = values * rank
                elif n_value_dims != rank:
                    raise ValueError("{} must be length 1, {} or {} but was {}.".format(name, rank, n_total_dims,
                        n_value_dims))

                values = [1] + values + [1]

            return values

        # strides
        strides = [1] * (rank + 2) if strides is None else reform(strides, 'strides')

        # dilations
        dilations = [1] * (rank + 2) if dilations is None else reform(dilations, 'dilations')

        if exists(mask): mask = tf.nn.pool(mask, kernels, 'MAX', strides=strides[1:-1], padding=padding,
                data_format=data_format, dilations=dilations[1:-1])

        # selection
        ops = gen_nn_ops.conv3d if rank == 3 else gen_nn_ops.conv2d

        if rank == 1:
            tensor = tf.expand_dims(tensor, axis=1)
            weight = weight[None, Ellipsis]
            strides = [strides[0], 1] + strides[1:]
            dilations = [dilations[0], 1] + dilations[1:]

        # perform operation
        tensor = ops(tensor, weight, strides, padding, use_cudnn_on_gpu=True, data_format='NHWC', dilations=dilations)
        if rank == 1: tensor = tf.squeeze(tensor, axis=[1])

        # bias
        if bias: tensor += bias_(([1]*(rank + 1) + [filters]), dtype, lrmul=lrmul, function=bias_function, trainable=trainable)

        # recover shape
        recover_shape = shape(tensor)

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0, rank + 1] + list(range(1, rank + 1)))
            batch_extend = tensor_shape[:n_batch_dims] + [filters] + recover_shape[1:-1]
        else:
            batch_extend = tensor_shape[:n_batch_dims] + recover_shape[1:]

        mask = reconstruct_mask(filters, mask=mask, axis=1 if data_format == 'channels_first' else -1)

        tensor = tf.reshape(tensor, batch_extend)
        if exists(mask):
            mask = tf.reshape(mask, batch_extend)
            tensor *= mask

    return tensor, mask



def deconvolution(
        tensor,
        rank,
        filters,
        kernels,
        strides=1,
        dilations=1,
        padding='SAME',
        groups=1,
        mask=None,
        bias=True,
        lrmul=1.0,
        quantization=0.0,
        quantization_blocks=8,
        weight_function=None,
        bias_function=None,
        data_format=None,
        trainable=True,
        scope='convolution'
        ):
    """
    convolution of <rank>-dimensional
    """
    with tf.variable_scope(scope):
        assert len(shape(tensor)) > rank + 1

        # padding check
        padding = padding.upper()

        # rank check
        if rank not in {1, 2, 3}:
            raise ValueError('The number of spatial dimensions must be one of 1, 2 or 3 but saw {}.'.format(rank))

        # filters check
        if isinstance(filters, float): filters = int(filters)
        if exists(filters) and filters % groups != 0:
            raise ValueError('The number of filters must be evenly divisible by the number of groups.'
                             'Received: groups={}, filters={}.'.format(groups, filters))

        # channels check
        dtype = tensor.dtype
        tensor_shape = shape(tensor)
        if tensor_shape[-1] % groups != 0:
            raise ValueError('The number of input channels must be evenly divisible by the number of groups.'
                             'Received groups={}, but the input has {} channels (full input shape is {}).'.format(groups, tensor_shape[-1], tensor_shape))

        # kernel size control
        if isinstance(kernels, int): kernels = [kernels, ] * rank
        kernels = list(kernels)
        if len(kernels) != rank:
            raise ValueError('The `kernels` argument must be a list of {} integers.'
                             'Received: {}.'.format(rank, kernels))
        for single_size in kernels:
            assert isinstance(single_size, int)
        if not all(kernels):
            raise ValueError('`kernels` cannot contain 0(s).'
                             'Received: {}'.format(kernels))

        # internal convolution operation
        n_total_dims = len(tensor_shape)
        n_batch_dims = n_total_dims - rank - 1
        batch_dims = list(range(0, n_batch_dims))

        # mask
        if exists(mask): tensor = tensor * mask

        weight = weight_(kernels + [filters//groups, tensor_shape[-1]//groups], dtype, lrmul=lrmul,
                function=weight_function, trainable=trainable)
        if trainable: weight = quantization_noise(weight, filters//groups, tensor_shape[-1], quantization, quantization_blocks)
        if groups > 1: weight = tf.tile(weight, [1] * rank + [1, groups])

        # manufacture shape
        tensor = tf.reshape(tensor, [-1] + tensor_shape[n_batch_dims:])
        if exists(mask): mask = tf.reshape(mask, [-1] + tensor_shape[n_batch_dims:])

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0] + list(range(2, rank + 2)) + [1])
            if exists(mask): mask = tf.transpose(mask, [0] + list(range(2, rank + 2)) + [1])

        def reform(values, name='values'):
            if isinstance(values, int): values = [values, ] * rank
            values = list(values)

            for single_size in values: assert isinstance(single_size, int)

            if not all(values): raise ValueError('`{}` cannot contain 0(s). Received: {}'.format(name, values))

            n_value_dims = len(values)

            if n_value_dims != (rank + 2):
                if n_value_dims == 1:
                    values = values * rank
                elif n_value_dims != rank:
                    raise ValueError("{} must be length 1, {} or {} but was {}.".format(name, rank, n_total_dims,
                        n_value_dims))

                values = [1] + values + [1]

            return values

        # strides
        strides = [1] * (rank + 2) if strides is None else reform(strides, 'strides')

        # dilations
        dilations = [1] * (rank + 2) if dilations is None else reform(dilations, 'dilations')

        # calculating output shape
        output_shape = shape(tensor)[:1]

        valid_spatial = lambda l, k, s, d: (l - 1) * s + k + (k-1) * (d-1)
        if padding == 'VALID':
            spatials = list(map(valid_spatial, shape(tensor)[1:-1], kernels, strides[1:-1], dilations[1:-1]))
        else:
            spatials = list(map(lambda l, s: l * s, shape(tensor)[1:-1], strides[1:-1]))

        output_shape += spatials + [filters]

        if exists(mask):
            mask_shape = shape(mask)
            f_, *mask_spatials, b_ = mask_shape
            mask_strides = strides.copy()
            for n in range(rank): mask_strides.insert(2 * n + 1, 1)
            for n in range(rank): mask_spatials.insert(2 * n + 1, 1)
            mask = tf.reshape(mask, [f_, *mask_spatials, b_])

            if padding == 'SAME':
                mask = tf.reshape(tf.tile(mask, mask_strides), list(map(lambda a, b: a * b, mask_shape, strides)))
            else:
                mask = tf.slice(mask, [0] * (len(mask_spatials) + 2), list(map(lambda s: tf.maximum(s - 1, 1), [f_, *mask_spatials, b_])))
                mask = tf.reshape(tf.tile(mask, mask_strides), list(map(lambda a, b: (a - 1) * b, mask_shape, strides)))
                pad_values = list(zip((np.asarray(kernels) - 1) * (np.asarray(dilations[1:-1]) - 1), np.asarray(kernels)))
                mask = tf.pad(mask, ((0, 0), *pad_values, (0, 0)))

        # selection and perform operation
        ops = gen_nn_ops.conv3d_backprop_input_v2 if rank == 3 else gen_nn_ops.conv2d_backprop_input

        if rank == 1:
            tensor = tf.expand_dims(tensor, axis=1)
            weight = weight[None, Ellipsis]
            strides = [strides[0], 1] + strides[1:]
            dilations = [dilations[0], 1] + dilations[1:]
            output_shape = [output_shape[0], 1] + output_shape[1:]

        tensor = ops(output_shape, weight, tensor, strides, padding, use_cudnn_on_gpu=True, dilations=dilations)
        if rank == 1: tensor = tf.squeeze(tensor, axis=[1])

        # bias
        if bias: tensor += bias_(([1]*(rank + 1) + [filters]), dtype, lrmul=lrmul, function=bias_function, trainable=trainable)

        # recover shape
        recover_shape = shape(tensor)

        if data_format == 'channels_first':
            tensor = tf.transpose(tensor, [0, rank + 1] + list(range(1, rank + 1)))
            if exists(mask): mask = tf.transpose(mask, [0, rank + 1] + list(range(1, rank + 1)))
            batch_extend = tensor_shape[:n_batch_dims] + [filters] + recover_shape[1:-1]
        else:
            batch_extend = tensor_shape[:n_batch_dims] + recover_shape[1:]

        mask = reconstruct_mask(filters, mask=mask, axis=1 if data_format == 'channels_first' else -1)

        tensor = tf.reshape(tensor, batch_extend)
        if exists(mask):
            mask = tf.reshape(mask, batch_extend)
            tensor *= mask

    return tensor, mask



def feed_forward(
        tensor,
        features,
        hidden_features=None,
        mask=None,
        activation=gelu,
        dropout_rate=0.,
        lrmul=1.0,
        quantization=0.0,
        weight_function=None,
        bias_function=None,
        trainable=True,
        scope='feed_fwd'
        ):
    with tf.variable_scope(scope):
        if hidden_features is None: hidden_features = 4 * features

        tensor, mask = linear(tensor, out_features=hidden_features, mask=mask, lrmul=lrmul, quantization=quantization,
                quantization_blocks=shape(tensor)[-1], weight_function=weight_function, bias_function=bias_function,
                trainable=trainable, scope='fc0')

        tensor = activation(tensor)
        if mask is not None: tensor *= mask

        if dropout_rate > 0.: tensor = tf.nn.dropout(tensor, rate=dropout_rate)

        tensor, mask = linear(tensor, out_features=features, mask=mask, lrmul=lrmul, quantization=quantization,
                quantization_blocks=hidden_features, weight_function=weight_function, bias_function=bias_function,
                trainable=trainable, scope='fc1')

        if dropout_rate > 0.: tensor = tf.nn.dropout(tensor, rate=dropout_rate)

        return tensor, mask



def attention(
        query,
        key,
        value,
        out_features=None,
        hidden_features=None,
        heads=4,
        mask_query=None,
        mask_key=None,
        mask_value=None,
        saved_state=None,
        state_distance=None,
        position_embedding=rotary_embedding,
        sequential_bias=None,
        projection_bias=True,
        dropout=0.0,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        query_weight_function=None,
        query_bias_function=None,
        key_weight_function=None,
        key_bias_function=None,
        value_weight_function=None,
        value_bias_function=None,
        out_weight_function=None,
        out_bias_function=None,
        trainable=True,
        scope='attention'
        ):
    with tf.variable_scope(scope):
        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        if out_features is None: out_features = q_shape[-1]
        if hidden_features is None: hidden_features = out_features
        hidden_depth = hidden_features // heads


        # linear layers for query, key, value : B, S, C --> B, S, H, D
        q, q_m = linear(query, in_features=q_shape[-1], out_features=[heads, hidden_depth], mask=mask_query,
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=query_weight_function, bias_function=query_bias_function, trainable=trainable,
                scope='projection_query')
        k, k_m = linear(key, in_features=k_shape[-1], out_features=[heads, hidden_depth], mask=mask_key,
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=key_weight_function, bias_function=key_bias_function, trainable=trainable,
                scope='projection_key')
        v, v_m = linear(value, in_features=v_shape[-1], out_features=[heads, hidden_depth], mask=mask_value,
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=value_weight_function, bias_function=value_bias_function, trainable=trainable,
                scope='projection_value')

        if q_m is None: q_m = tf.ones_like(q)
        if k_m is None: k_m = tf.ones_like(k)
        if v_m is None: v_m = tf.ones_like(v)

        # saved states for key, value
        if saved_state is not None:
            k_s, v_s, k_m_s, v_m_s = saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_state["value_mask"]
            k_m = tf.concat([tf.ones_like(k_s) if k_m_s is None else k_m_s, k_m], axis=1)
            v_m = tf.concat([tf.ones_like(v_s) if v_m_s is None else v_m_s, v_m], axis=1)

            k, v = tf.concat([k_s, k], axis=1), tf.concat([v_s, v], axis=1)
            sort_k, sort_v = tf.argsort(k_m, axis=1), tf.argsort(v_m, axis=1)
            k, v = tf.gather(k, sort_k, batch_dims=1), tf.gather(v, sort_v, batch_dims=1)
            if isinstance(state_distance, int):
                cut_k, cut_v = tf.maximum(0, shape(k)[1]-state_distance), tf.maximum(0, shape(v)[1] - state_distance)
                k, v, k_m, v_m = k[cut_k:], v[cut_v:], k_m[cut_k:], v_m[cut_v:]

            saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_state["value_mask"] = k, v, k_m, v_m

        # position embedding
        if exists(position_embedding):
            q = position_embedding(q, q_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_q')
            k = position_embedding(k, k_shape[-2], hidden_depth, trainable=trainable, scope='pos_emb_k')

        # score weights
        logits = tf.reduce_sum(q[:, :, None] * tf.rsqrt(tf.to_float(hidden_depth)) * k[:, None], axis=-1)     # bqkh
        if sequential_bias is not None: logits += sequential_bias
        logits -= tf.reduce_max(logits, axis=-1, keepdims=True)

        # masked softmax
        attn_bm = tf.equal(tf.reduce_max(q_m, axis=-1)[:, :, None] * tf.reduce_max(k_m, axis=-1)[:, None], 0)     # bqkh
        attn_weight = tf.where(attn_bm, tf.zeros_like(logits), tf.math.exp(logits))
        attn_weight = tf.where(attn_bm, tf.zeros_like(logits), attn_weight / tf.maximum(tf.reduce_sum(attn_weight, axis=2, keepdims=True), 1e-8))
        if trainable: attn_weight = tf.nn.dropout(attn_weight, rate=dropout)

        # attention operation
        tensor = tf.reduce_sum(attn_weight[Ellipsis, None] * v[:, None], axis=2)     # bqhd


        # linear layer for output
        tensor, mask_tensor = linear(tensor, in_features=[heads, hidden_depth], out_features=out_features, mask=q_m,
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                scope='projection_out')

        if mask_query is None: mask_tensor = None

        return tensor, mask_tensor, attn_weight, saved_state



def upsampolate(
        tensor,
        features,
        axis=1,
        up_rate=2,
        mask=None,
        ):
    # strictrly required form of, single batch size B frontmost, and single channels C backmost
    t_shape = shape(tensor)
    max_axis = len(t_shape) - 1
    if axis < 0: axis += len(t_shape)
    assert 0 < axis and axis < max_axis

    tensor_tile = tf.tile(tf.expand_dims(tensor, axis + 1), [1] * (axis + 1) + [up_rate] + [1] * (max_axis - axis))
    tensor_expand = tf.reshape(tensor_tile, t_shape[:axis] + [up_rate * t_shape[axis]] + t_shape[axis + 1:])
    kernel_size = 2 * up_rate
    pad_left, pad_right = (kernel_size - 1) // 2, kernel_size // 2
    tensor_pad = tf.concat([tf.gather(tensor, [0] * pad_left, axis=axis, batch_dims=0), tensor_expand,
        tf.gather(tensor, [t_shape[axis] - 1] * pad_right, axis=axis, batch_dims=0)], axis=axis)
    filters = tf.ones([kernel_size, t_shape[-1], features], dtype=tensor.dtype) / (kernel_size * t_shape[-1])
    if max_axis == 2: # 1d
        upsampolated = tf.nn.conv1d(tensor_pad, filters, stride=1, padding='VALID')
    elif max_axis == 3: # 2d, axis=1 or axis=2
        upsampolated = tf.nn.conv2d(tensor_pad, tf.expand_dims(filters, 2 - axis), strides=(1, 1), padding='VALID')
    else:
        raise ValueError("1d and 2d only implemented")

    if exists(mask):
        mask_tile = tf.tile(tf.expand_dims(mask, axis + 1), [1] * (axis + 1) + [up_rate] + [1] * (max_axis - axis))
        mask_expand = tf.reshape(mask_tile, t_shape[:axis] + [up_rate * t_shape[axis]] + t_shape[axis + 1:])
        mask = tf.tile(tf.reduce_max(mask_expand, axis=-1, keepdims=True), [1] * max_axis + [features])
        upsampolated *= mask

    return upsampolated, mask



def favorplus(
        query,
        key,
        value,
        out_features=None,
        hidden_features=None,
        map_features=None,
        heads=4,
        mask_query=None,
        mask_key=None,
        mask_value=None,
        input_relation='self',
        map_projection=True,
        map_function='relu',
        seed_value=None,
        use_recency_bias=False,
        saved_state=None,
        state_distance=None,
        #sequential_bias=None,
        projection_bias=True,
        #dropout_rate=0.0,
        quantization=0.0,
        quantization_blocks=8,
        lrmul=1.0,
        query_weight_function=None,
        query_bias_function=None,
        key_weight_function=None,
        key_bias_function=None,
        value_weight_function=None,
        value_bias_function=None,
        out_weight_function=None,
        out_bias_function=None,
        trainable=True,
        scope='favorplus'
        ):
    with tf.variable_scope(scope):
        if not isinstance(heads, int): raise ValueError("The number of heads must be integer,"
                " but given {}".format(type(heads)))
        q_shape, k_shape, v_shape = shape(query), shape(key), shape(value)
        dtype = query.dtype
        if not isinstance(heads, int): heads = 1
        if not isinstance(out_features, int): out_features = q_shape[-1]
        if out_features % heads != 0: raise ValueError("The number of heads must divide out_units evenly,"
                " but heads:{} and out_units: {}".format(heads, out_features))
        if not isinstance(hidden_features, int): hidden_features = out_features
        if hidden_features % heads != 0: raise ValueError("The number of heads must divide hidden_units evenly,"
                " but heads:{} and hidden_units: {}".format(heads, hidden_features))
        hidden_depth = hidden_features // heads
        theta_random_projection_features = np.ceil(hidden_depth * np.log(hidden_depth)).astype(np.int32)
        if isinstance(map_features, int):
            theta_random_projection_features = max(map_features, theta_random_projection_features)

        if not input_relation in ['self', 'causal', 'cross']:
            raise ValueError("input relation must be one of `self`, `causal`, `cross`.")
        if not map_function in ['relu', 'softmax']:
            raise NotImplementedError("given method {} of shift-invariant kernels is unsupported.".format(map_function))

        use_recency_bias = use_recency_bias and input_relation == 'self'

        B = tf.reduce_prod(q_shape[:-2], keepdims=False)

        # material projection
        q, q_m = linear(query, out_features=[heads, hidden_depth], mask=mask_query, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=query_weight_function, bias_function=query_bias_function, trainable=trainable,
                scope='projection_query')
        k, k_m = linear(key, out_features=[heads, hidden_depth], mask=mask_key, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=key_weight_function, bias_function=key_bias_function, trainable=trainable,
                scope='projection_key')
        v, v_m = linear(value, out_features=[heads, hidden_depth], mask=mask_value, bias=projection_bias, lrmul=lrmul,
                quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=value_weight_function, bias_function=value_bias_function, trainable=trainable,
                scope='projection_value')

        if q_m is None: q_m = tf.ones_like(q)
        if k_m is None: k_m = tf.ones_like(k)
        if v_m is None: v_m = tf.ones_like(v)

        # map projection
        def _create_products_of_givens_rotations(d, seed=None):
            n_givens_rotations = d * int(math.ceil(math.log(float(d))))
            q = np.eye(d, d)
            if seed is not None: np.random.seed(seed)
            for _ in range(n_givens_rotations):
                random_angle = math.pi * np.random.uniform()
                random_indices = np.random.choice(d, 2)
                index_i = min(random_indices[0], random_indices[1])
                index_j = max(random_indices[0], random_indices[1])
                slice_i = q[index_i]
                slice_j = q[index_j]
                new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
                new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
                q[index_i] = new_slice_i
                q[index_j] = new_slice_j
            return tf.cast(tf.constant(q), dtype=dtype)

        def _create_projection_matrix(m, d, seed, scaling=0, struct_mode=False):
            n_full_blocks = m // d
            block_list = []
            current_seed = seed
            for _ in range(n_full_blocks):
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q)
                current_seed += 1
            remaining_rows = m - n_full_blocks * d
            if remaining_rows > 0:
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q[0:remaining_rows])
            final_matrix = tf.concat(block_list, axis=0)
            current_seed += 1

            if scaling == 0:
                multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
            elif scaling == 1:
                multiplier = tf.math.sqrt(float(d)) * tf.ones(float(m))
            else:
                raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)
            return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)

        if map_projection:
            # seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(Q) * 1e8))
            # seed = tf.dtypes.cast(seed, tf.int32)
            # projection_matrix = _create_projection_matrix(seed=seed)
            seed = random.randint(0, (2 ** 31) - 1) if seed_value is None else seed_value
            projection_matrix = _create_projection_matrix(theta_random_projection_features, hidden_depth, seed=seed)
        else:
            projection_matrix = None

        if use_recency_bias:
            gate, gate_m = linear(key, out_features=heads, mask=mask_key, bias=True, lrmul=lrmul,
                    trainable=trainable, scope='projection_gate')
            gate = tf.math.sigmoid(gate)
            if gate_m is not None: gate *= gate_m
        else:
            gate = None

        # cache
        if saved_state is not None:
            k_s, v_s, k_m_s, v_m_s = saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_state["value_mask"]
            k_m = tf.concat([tf.ones_like(k_s) if k_m_s is None else k_m_s, k_m], axis=1)
            v_m = tf.concat([tf.ones_like(v_s) if v_m_s is None else v_m_s, v_m], axis=1)

            k, v = tf.concat([k_s, k], axis=1), tf.concat([v_s, v], axis=1)
            sort_k, sort_v = tf.argsort(k_m, axis=1), tf.argsort(v_m, axis=1)
            k, v = tf.gather(k, sort_k, batch_dims=1), tf.gather(v, sort_v, batch_dims=1)
            if isinstance(state_distance, int):
                cut_k, cut_v = tf.maximum(0, shape(k)[1]-state_distance), tf.maximum(0, shape(v)[1] - state_distance)
                k, v, k_m, v_m = k[cut_k:], v[cut_v:], k_m[cut_k:], v_m[cut_v:]

            saved_state["key"], saved_state["value"], saved_state["key_mask"], saved_state["value_mask"] = k, v, k_m, v_m

            if use_recency_bias:
                gate_s, gate_m_s = saved_state["gate"], saved_state["gate_mask"]
                gate_m = tf.concat([tf.ones_like(gate_s) if gate_m_s is None else gate_m_s, gate_m], axis=1)

                gate = tf.concat([gate_s, gate], axis=1)
                gate = tf.gather(gate, sort_k, batch_dims=1)
                if isinstance(state_distance, int):
                    gate, gate_m = gate[cut_k:], gate_m[cut_k:]

                saved_state["gate"], saved_state["gate_mask"] = gate, gate_m

            input_relation = 'cross'

        def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):

            del is_query
            if projection_matrix is None:
                return tf.nn.relu(data) + numerical_stabilizer
            else:
                ratio = 1.0 * tf.math.rsqrt(tf.sqrt(tf.cast(theta_random_projection_features, tf.float32)))
                data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
                return tf.nn.relu(data_dash) + numerical_stabilizer

        def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):

            data_normalizer = tf.math.rsqrt(tf.math.sqrt(tf.cast(shape(data)[-1], tf.float32)))
            data = data_normalizer * data

            ratio = tf.math.rsqrt(tf.cast(shape(data)[0], tf.float32))
            data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
            diag_data = tf.math.square(data)
            diag_data = tf.math.reduce_sum(diag_data, axis=-1)
            diag_data = diag_data / 2.0
            diag_data = tf.expand_dims(diag_data, axis=-1)
            last_dims_t = (3,)
            attention_dims_t = (2,)
            if is_query:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=-1, keepdims=True))
                    + numerical_stabilizer)
            else:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=[-3, -1], keepdims=True))
                    + numerical_stabilizer)
            return data_dash

        if map_function == 'relu':
            kernel_function = relu_kernel_transformation
        elif map_function == 'softmax':
            kernel_function = softmax_kernel_transformation
        else:
            raise NotImplementedError("Not Implemented Yet")

        q_prime = tf.transpose(kernel_function(q, True, projection_matrix), [1, 0, 2, 3])
        k_prime = tf.transpose(kernel_function(k, False, projection_matrix), [1, 0, 2, 3])
        v_prime = tf.transpose(v, [1, 0, 2, 3])
        if use_recency_bias: gate = tf.transpose(gate, [1, 0, 2])

        @tf.custom_gradient
        def causal_numerator(qs, ks, vs, gate=None):
            result = []
            sums = tf.zeros([B, heads, theta_random_projection_features, hidden_depth], dtype=dtype)
            if gate is None:
                for i_th in range(q_shape[-2]):
                    sums = sums + tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(q_shape[-2]):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                grads = tf.zeros([B, heads, theta_random_projection_features, hidden_depth], dtype=dtype)

                gradient_sums = sums

                q_grads = []
                k_grads = []
                v_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(q_shape[-2] - 1, -1, -1):
                        q_grads.append(tf.einsum("bhmo,bho->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo", qs[i_th], res_grad[i_th])
                        k_grads.append(tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[None, Ellipsis])
                        v_grads.append(tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[None, Ellipsis])
                        gradient_sums = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads

                else:
                    for i_th in range(q_shape[-2] - 1, -1, -1):
                        q_grads.append(tf.einsum("bhmo,bho->bhm",
                            gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo",
                                qs[i_th], res_grad[i_th]) # (1 - gate[i_th]) * bhmo(from bhm * bho) + gate[i_th] * sums[i_th-1]

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]  # bhm
                        v_grads_gate = tf.gradients(gate[i_th], vs[i_th])[0]  # bho

                        k_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[None, Ellipsis]
                        v_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[None, Ellipsis]

                        gradient_sums = gradient_sums - ((1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                        gradient_sums = gradient_sums / tf.math.maximum(gate[i_th], 1e-8)

                        sums_before_minus_curr_cross = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                        if k_grads_gate is None:
                            k_grads.append(k_grads_k_cross_v)
                        else:
                            k_grads_with_gate = k_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-1)
                            # TODO not sure that k_grads_gate has shape of bho, and not sure einsum is sufficient here
                            k_grads.append(k_grads_with_gate + k_grads_k_cross_v)

                        if v_grads_gate is None:
                            v_grads.append(v_grads_k_cross_v)
                        else:
                            v_grads_with_gate = v_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-2)
                            # TODO not sure that v_grads_gate has shape of bhm, and not sure einsum is sufficient here
                            v_grads.append(v_grads_with_gate + v_grads_k_cross_v)

                        gate_grads_term = tf.gradients(tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]), gate[i_th])[0]
                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_cross)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term + tf.reduce_sum(sums_before_minus_curr_cross,
                                axis=[-2, -1])[Ellipsis, None])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads, gate_grads

            return result, grad

        @tf.custom_gradient
        def causal_denominator(qs, ks, gate=None):
            result = []
            sums = tf.zeros([B, heads, theta_random_projection_features])
            if gate is None:
                for i_th in range(q_shape[-2]):
                    sums = sums + ks[i_th]
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(q_shape[-2]):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * ks[i_th])
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                z_grad = tf.zeros([B, heads, theta_random_projection_features])

                gradient_sums = sums

                q_grads = []
                k_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(q_shape[-2] - 1, -1, -1):
                        q_grads.append(tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])
                        k_grads.append(z_grad[None, Ellipsis])
                        gradient_sums = gradient_sums - ks[i_th]

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], aixs=0)

                    return q_grads, k_grads

                else:
                    for i_th in range(q_shape[-2] - 1, -1, -1):
                        q_grads.append(tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]

                        gradient_sums = gradient_sums - ((1.0 - gate[i_th]) * ks[i_th])
                        gradient_sums = gradient_sums / gate[i_th]

                        sums_before_minus_curr_k = gradient_sums - ks[i_th]

                        if k_grads_gate is None:
                            k_grads.append((1.0 - gate[i_th]) * z_grad)
                        else:
                            k_grads.append(k_grads_gate * sums_before_minus_curr_k + z_grad)
                        gate_grads_term = tf.gradients(ks[i_th], gate[i_th])[0]

                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_k)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term + sums_before_minus_curr_k)

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, gate_grads

            return result, grad

        def noncausal_numerator(qs, ks, vs, gate=None):
            if gate is None:
                kvs = tf.einsum("lbhm,lbho->bhmo", ks, vs)
                return tf.einsum("Lbhm,bhmo->Lbho", qs, kvs)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                kvs = tf.einsum("lbhm,lbho->lbhmo", ks, vs)
                remainder_kvs = tf.einsum("lbh,lbhmo->bhmo", reverse_gate, kvs)
                fully_gated_kvs = tf.einsum("lbh,bhmo->lbhmo", gate, remainder_kvs) - tf.einsum("lbh,lbhmo->lbhmo",
                        tf.math.square(reverse_gate), kvs)
                return tf.einsum("Lbhm,Lbhmo->Lbho", qs, fully_gated_kvs)

        def noncausal_denominator(qs, ks, gate=None):
            if gate is None:
                ks_sum = tf.einsum("lbhm,l->bhm", ks, tf.ones([k_shape[-2]], dtype=dtype))
                return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                remainder_ks = tf.einsum("lbh,lbhm->bhm", reverse_gate, ks)
                fully_gated_ks = tf.einsum("lbh,bhm->lbhm", gate, remainder_ks) - tf.einsum("lbh,lbhm->lbhm",
                        tf.math.square(reverse_gate), ks)
                return tf.einsum("Lbhm,Lbhm->Lbh", qs, fully_gated_ks)

        if input_relation == 'causal':
            av_attention = causal_numerator(q_prime, k_prime, v_prime, gate=gate)
            attention_normalizer = causal_denominator(q_prime, k_prime, gate=gate)
        else:
            av_attention = noncausal_numerator(q_prime, k_prime, v_prime, gate=gate)
            attention_normalizer = noncausal_denominator(q_prime, k_prime, gate=gate)

        attention_normalizer = tf.tile(attention_normalizer[Ellipsis, None], [1, 1, 1, hidden_depth])
        tensor = tf.where(tf.equal(attention_normalizer, 0.0), tf.zeros_like(av_attention), av_attention / attention_normalizer)
        tensor = tf.transpose(tensor, [1, 0, 2, 3])

        tensor, mask_tensor = linear(tensor, in_features=shape(tensor)[-2:], out_features=out_features,
                mask=tf.tile(tf.reduce_max(q_m, axis=[-2, -1], keepdims=True), [1, 1] + shape(tensor)[-2:]),
                bias=projection_bias, lrmul=lrmul, quantization=quantization, quantization_blocks=quantization_blocks,
                weight_function=out_weight_function, bias_function=out_bias_function, trainable=trainable,
                scope='projection_out')

        tensor = tf.reshape(tensor, q_shape[:-1] + [out_features])
        if mask_query is None: mask_tensor = None

    return tensor, mask_tensor, saved_state



##### TODO fix down #####
def window_self_attention_2d(
        tensor,
        num_units,
        num_heads,
        window_size=(7,7),
        pretrained_window_size=(0,0),
        shift_size=(0,0),
        mask=None,
        dropout_rate=.4,
        qkv_bias=True,
        proj_bias=True,
        weight_normalizer=None,
        trainable=True,
        scope='window_attention'
        ):
    '''
    input tensors has shape of
        [batch, height, width, channels]
    mask has shape of
        [batch, height, width, channels]
    if not None
    '''
    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        B, H, W, C = shape(tensor)
        dtype = tensor.dtype
        depth = num_units // num_heads
        if isinstance(shift_size, int): shift_size = (shift_size, shift_size)
        if isinstance(shift_size, list): shift_size = tuple(shift_size)
        assert len(shift_size) == 2
        mask_was_None = mask is None
        if mask_was_None:
            mask = tf.ones_like(tensor, dtype=dtype)
        mask = tf.tile(tf.reduce_min(mask, axis=-1, keepdims=True), [1, 1, 1, C])
        tensor = tensor * mask

        winH, winW = window_size
        pret_winH, pret_winW = pretrained_window_size
        sftH, sftW = shift_size
        #with tf.control_dependencies([tf.debugging.Assert(winH < H, [winH, H]),
        #    tf.debugging.Assert(winW < W, [winW, W])]):
        #    H, W = list(map(lambda in_: tf.identity(in_), [H, W]))

        if shift_size != (0, 0):
            tensor = tf.roll(tensor, shift=(-sftH, -sftW), axis=(1,2))

        '''partition begin'''
        # unitization
        padH, padW = winH - (H % winH), winW - (W % winW)
        unitized_tensor_mask = list(map(lambda arr: tf.pad(
            arr, [[0,0], [0,padH], [0,padW], [0,0]], mode='CONSTANT', constant_values=0.),
            [tensor, mask]))
        _, H_unit, W_unit, _ = shape(unitized_tensor_mask[0]) # tensor shape

        # reform
        refH, refW = H_unit//winH, W_unit//winW
        reformed_tensor_mask = list(map(lambda arr: tf.reshape(tf.transpose(
            tf.reshape(arr, [B, refH, winH, refW, winW, C]), [0, 1, 3, 2, 4, 5]),
            [-1, winH, winW, C]), unitized_tensor_mask))
        '''partition end'''

        reformed, mask_reformed = list(map(lambda arr: tf.reshape(
            arr, [-1, winH*winW, C]), reformed_tensor_mask))

        '''relative position embedding begin'''
        # relative position bias
        rel_coords_h_tile_w = tf.tile(tf.range(-(winH-1), winH, dtype=dtype,
            name='rel_coords_h')[:, None], [1, 2*winW-1])
        rel_coords_w_tile_h = tf.tile(tf.range(-(winW-1), winW, dtype=dtype,
            name='rel_coords_w')[None], [2*winH-1, 1])
        rel_coords_table = tf.stack([rel_coords_h_tile_w, rel_coords_w_tile_h], axis=0)
        rel_coords_table = tf.transpose(rel_coords_table, [1, 2, 0])[None]
        divisor = tf.convert_to_tensor([pret_winH-1, pret_winW-1] if pret_winH > 0\
                else [winH-1, winW-1], dtype=dtype)
        rel_coords_table /= divisor[None, None]
        rel_coords_table = tf.math.sign(rel_coords_table*8) * tf.math.log1p(
                tf.math.abs(rel_coords_table)) / np.log(8)

        idx_h_tile_w = tf.tile(tf.range(winH, dtype=tf.int32)[:, None], [1, winW])
        idx_w_tile_h = tf.tile(tf.range(winW, dtype=tf.int32)[None], [winH, 1])
        coords = tf.stack([idx_h_tile_w, idx_w_tile_h], axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])
        rel_coords = tf.transpose(
                coords_flatten[Ellipsis, None] - coords_flatten[:, None],
                [1, 2, 0])
        rel_coords += tf.convert_to_tensor([winH-1, winW-1], dtype=tf.int32)[None, None]
        rel_coords = tf.stack([rel_coords[Ellipsis, 0] * (2*winW-1),
            rel_coords[Ellipsis, 1]], axis=-1)
        rel_pos_idx = tf.reduce_sum(rel_coords, axis=-1)
        '''relative position embedding end'''

        # linear layers for query, keys, values : B, S, C --> B, S, H, D
        qkv_weight = tf.get_variable('qkv_linear', [C, num_heads, depth*3], dtype=dtype,
                initializer=kaiming_uniform_init(C*num_heads), trainable=trainable)
        if weight_normalizer: qkv_weight = weight_normalizer(qkv_weight, name='w_qkv')
        qkv = tf.einsum('bwc,cnd->bnwd', reformed, qkv_weight)
        if qkv_bias: qkv += tf.get_variable('qkv_bias', [num_heads, depth*3], dtype=dtype,
                initializer=tf.zeros_initializer, trainable=trainable)[None, :, None]
        mask_qkv = tf.reduce_min(mask_reformed, axis=-1, keepdims=True)[:, None]
        qkv *= mask_qkv
        query, keys, values = tf.split(qkv, 3, axis=-1)

        # score weights
        logits = tf.einsum('bnhd,bnwd->bnhw', p_normalize(query), p_normalize(keys))

        # logit scale
        logit_scale = tf.get_variable('logit_scale', [num_heads, 1, 1], dtype=dtype,
                initializer=tf.constant_initializer(np.log(10.)),
                trainable=trainable)
        logit_scale = tf.where(logit_scale > 100.,
                100.*tf.ones_like(logit_scale), logit_scale)[None]
        logits *= logit_scale

        with tf.variable_scope('continuous_rel_pos_bias'):
            rel_pos_bias_table = linear(tf.nn.relu(linear(
                rel_coords_table, 512, trainable=trainable, scope='linear_0')[0]),
                num_heads, bias=False, trainable=trainable, scope='linear_1')[0]
        rel_pos_bias_table = tf.reshape(rel_pos_bias_table, [-1, num_heads])
        rel_pos_bias = tf.transpose(tf.gather(rel_pos_bias_table, rel_pos_idx, axis=0),
                [2, 0, 1])
        rel_pos_bias = 16. * tf.math.sigmoid(rel_pos_bias)
        logits += rel_pos_bias[None]

        # window shifting mask
        if shift_size != (0, 0):
            mask_shift = tf.zeros((H_unit, W_unit), dtype)
            h_slices = (slice(0, -winH), slice(-winH, -sftH), slice(-sftH, None))
            w_slices = (slice(0, -winW), slice(-winW, -sftW), slice(-sftW, None))
            cnt = 0
            tops = tf.concat([
                tf.zeros([W_unit-winW], dtype=dtype),
                tf.ones([winW-sftW], dtype=dtype),
                2.*tf.ones([sftW], dtype=dtype)],
                axis=0)
            mids = tops + 3.
            bots = tops + 6.
            mask_shift = tf.concat([
                tf.tile(tops[None], [H_unit-winH, 1]),
                tf.tile(mids[None], [winH-sftH, 1]),
                tf.tile(bots[None], [sftH, 1])], axis=0)
            mask_sw = tf.reshape(mask_shift, [refH, winH, refW, winW])
            mask_sw = tf.transpose(mask_sw, [0, 2, 1, 3])
            mask_sw = tf.reshape(mask_sw, [-1, winH * winW])
            mask_sw_attn = mask_sw[:, None] - mask_sw[Ellipsis, None]
            mask_sw_attn = tf.where(tf.equal(mask_sw_attn, 0.), tf.ones_like(mask_sw_attn),
                    tf.zeros_like(mask_sw_attn))

        # spatial mask
        mask_logits = tf.tile(mask_qkv, [1, num_heads, 1, 1])     # B*refH*refW, winH * winW
        mask_logits = mask_logits * tf.squeeze(mask_logits, axis=-1)[Ellipsis, None, :]
        if shift_size != (0, 0):
            mask_logits *= tf.tile(mask_sw_attn[:, None], [B, 1, 1, 1])
        bool_mask_zero = tf.equal(mask_logits, 0.)
        logits -= tf.reduce_max(logits, axis=-1, keepdims=True)

        weights = tf.where(bool_mask_zero, tf.zeros_like(logits), tf.math.exp(logits))
        weights /= tf.math.maximum(tf.reduce_sum(weights, axis=-1, keepdims=True), 1e-8)
        weights = tf.where(bool_mask_zero, tf.zeros_like(weights), weights)

        if trainable: weights = tf.nn.dropout(weights, rate=dropout_rate)

        # attention operation
        attention_windows = tf.einsum('bnxy,bnyd->bnxd', weights, values)

        # output dense
        output_weight = tf.get_variable('output_linear', [num_heads, depth, num_units], dtype=dtype,
                initializer=kaiming_uniform_init(num_heads*depth), trainable=trainable)
        if weight_normalizer: output_weight = weight_normalizer(output_weight, name='w_o')
        attention_windows = tf.einsum('bnwd,ndu->bwu', attention_windows, output_weight)
        if proj_bias: attention_windows += tf.get_variable('proj_bias', [num_units], dtype=dtype,
                initializer=tf.zeros_initializer, trainable=trainable)[None, None]

        # shape reconstruction
        attention_windows = tf.reshape(attention_windows, [-1, winH, winW, num_units])
        swin_attention = tf.reshape(
                attention_windows, [B, refH, refW, winH, winW, num_units])
        swin_attention = tf.reshape(tf.transpose(swin_attention, [0, 1, 3, 2, 4, 5]),
                [B, H_unit, W_unit, num_units])

        if shift_size != (0, 0):
            swin_attention = tf.roll(swin_attention, shift=(sftH, sftW), axis=(1,2))

        swin_attention_output = swin_attention[:, :-padH, :-padW, :]
        mask = tf.tile(tf.reduce_min(mask, axis=-1, keepdims=True),
                [1, 1, 1, num_units])
        swin_attention_output *= mask

        return swin_attention_output, None if mask_was_None else mask


##### TODO fix down #####
def downsample_1d(
        tensor,
        channels,
        down_size=1,
        overlap_size=0,
        mask=None,
        bias=True,
        weight_normalizer=None,
        pad_values=0.0,
        trainable=True,
        scope='downsample_1d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, W, C = shape(tensor)
        filters = tf.get_variable('filters', [down_size+(2*overlap_size), C, channels], dtype=dtype,
                initializer=kaiming_uniform_init((down_size+(2*overlap_size))*C), trainable=trainable)
        if weight_normalizer: filters = weight_normalizer(filters)
        pW = -W % down_size
        tensor_pad = tf.pad(tensor, [[0, 0], [pW // 2 + overlap_size, (pW + 1) // 2 + overlap_size], [0, 0]],
                mode='CONSTANT', constant_values=pad_values)
        down = tf.nn.conv1d(tensor_pad, filters, stride=down_size, padding='VALID', data_format='NWC')
        if bias:
            down += tf.get_variable('bias', [channels], dtype=dtype, initializer=tf.zeros_initializer,
                    trainable=trainable)[None, None]
        if mask is not None:
            mask = tf.pad(mask, [[0, 0], [pW // 2, (pW + 1) // 2], [0, 0]], mode='CONSTANT', constant_values=0)
            mask = tf.tile(tf.reduce_max(mask[:, 0::down_size, :], axis=-1, keepdims=True), [1, 1, channels])
            down *= mask
        return down, mask


##### TODO fix down #####
def upsample_1d(
        tensor,
        channels,
        up_size=1,
        overlap_size=0,
        mask=None,
        bias=True,
        weight_normalizer=None,
        trainable=True,
        scope='upsample_1d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, W, C = shape(tensor)
        output_shape = [B, up_size*W, channels]
        filters = tf.get_variable('filters', [up_size+(2*overlap_size), channels, C], dtype=dtype,
                initializer=kaiming_uniform_init((up_size+(2*overlap_size))*channels), trainable=trainable)
        if weight_normalizer: filters = weight_normalizer(filters)
        up = tf.nn.conv1d_transpose(tensor, filters, output_shape, strides=up_size, padding='SAME', data_format='NWC')
        if bias:
            up += tf.get_variable('bias', [channels], dtype=dtype, initializer=tf.zeros_initializer,
                    trainable=trainable)[None, None]
        if mask is not None:
            mask = tf.tile(tf.reduce_max(mask[:,:,None], axis=-1, keepdims=True), [1, 1, up_size, channels])
            mask = tf.reshape(mask, output_shape)
            up *= mask
        return up, mask


##### TODO fix down #####
def downsample_2d(
        tensor,
        channels,
        down_size=[1,1],
        overlap_size=[0,0],
        mask=None,
        bias=True,
        weight_normalizer=None,
        pad_values=0.0,
        trainable=True,
        scope='downsample_2d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, H, W, C = shape(tensor)
        dH, dW = down_size
        oH, oW = overlap_size
        filters = tf.get_variable('filters', [dH+(2*oH), dW+(2*oW), C, channels], dtype=dtype,
                initializer=kaiming_uniform_init((dH+(2*oH))*(dW+(2*oW))*C), trainable=trainable)
        if weight_normalizer: filters = weight_normalizer(filters)
        pH, pW = (-H % dH), (-W % dW)
        tensor_pad = tf.pad(tensor, [[0, 0], [pH // 2 + oH, (pH + 1) // 2 + oH], [pW // 2 + oW, (pW + 1) // 2 + oW], [0, 0]],
                mode='CONSTANT', constant_values=pad_values)
        down = tf.nn.conv2d(tensor_pad, filters, strides=down_size, padding='VALID', data_format='NHWC')
        if bias:
            down += tf.get_variable('bias', [channels], dtype=dtype, initializer=tf.zeros_initializer,
                    trainable=trainable)[None, None, None]
        if mask is not None:
            mask = tf.pad(mask, [[0, 0], [pH // 2, (pH + 1) // 2], [pW // 2, (pW + 1) // 2], [0, 0]],
                    mode='CONSTANT', constant_values=0)
            mask = tf.tile(tf.reduce_max(mask[:, 0::dH, 0::dW, :], axis=-1, keepdims=True), [1, 1, 1, channels])
            down *= mask
        return down, mask


##### TODO fix down #####
def upsample_2d(
        tensor,
        channels,
        up_size=[1,1],
        overlap_size=[0,0],
        mask=None,
        bias=True,
        weight_normalizer=None,
        trainable=True,
        scope='upsample_2d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, H, W, C = shape(tensor)
        uH, uW = up_size
        oH, oW = overlap_size
        output_shape = [B, uH*H, uW*W, channels]
        filters = tf.get_variable('filters', [uH+(2*oH), uW+(2*oW), channels, C], dtype=dtype,
                initializer=kaiming_uniform_init((uH+(2*oH))*(uW+(2*oW))*channels), trainable=trainable)
        if weight_normalizer: filters = weight_normalizer(filters)
        up = tf.nn.conv2d_transpose(tensor, filters, output_shape, strides=up_size, padding='SAME', data_format='NHWC')
        if bias:
            up += tf.get_variable('bias', [channels], dtype=dtype, initializer=tf.zeros_initializer,
                    trainable=trainable)[None, None, None]
        if mask is not None:
            mask = tf.tile(tf.reduce_max(mask[:,:,None,:,None], axis=-1, keepdims=True), [1, 1, uH, 1, uW, channels])
            mask = tf.reshape(mask, output_shape)
            up *= mask
        return up, mask



##### TODO fix down #####
def modulated_conv1d(
        tensor,
        channels,
        kernel_size,
        style=None,
        mask=None,
        demodulate=True,
        fused=True,
        padding='SAME',
        input_gain=None,
        epsilon=1e-8,
        bias=True,
        trainable=True,
        scope='mod_conv1d'):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, L, C_in = shape(tensor)
        K, C_out = kernel_size, channels
        padding=padding.upper()
        if padding == 'SAME':
            tensor = tf.pad(tensor, [[0,0], [(K-1)//2,K//2], [0,0]])
            padding = 'VALID'
            cutter = [0, 0]
        elif padding == 'VALID':
            cutter = [(K-1)//2, K//2]

        weight = tf.get_variable('weight', shape=[K, C_in, C_out], dtype=dtype,
                initializer=kaiming_uniform_init(K*C_in), trainable=trainable)
        ww = weight[None]

        if style is not None:
            style = linear(style, C_in, trainable=trainable, scope='linear')[0] + 1.
            ww *= tf.cast(style[:, None, :, None], dtype)

        if demodulate:
            demod = tf.math.rsqrt(tf.maximum(tf.einsum('blio,blio->bo', ww, ww), epsilon))
            ww *= demod[:, None, None]

        if input_gain is not None:
            ww *= tf.broadcast_to(input_gain, [shape(ww)[0],1,C_in,1])

        if style is None:
            ww = tf.squeeze(ww, axis=0) # lio
        elif fused:
            t_shape = [1,shape(tensor)[1],B*C_in]
            tensor = tf.reshape(tf.transpose(tensor, [1,0,2]), t_shape)
            ww = tf.reshape(tf.transpose(ww, [1,2,0,3]), [K,C_in,B*C_out])
        else:
            tensor *= tf.cast(style[:, None, :], dtype=dtype) # bli

        tensor = tf.nn.conv1d(tensor, ww, stride=1, padding=padding)

        if fused and style is not None:     # tensor : 1, L, B*C_out
            tensor = tf.transpose(tf.reshape(tensor, [shape(tensor)[1],B,C_out]),
                    [1,0,2])
        elif demodulate:
            tensor *= tf.cast(demod[:, None], dtype=dtype)

        if bias: tensor += tf.broadcast_to(tf.get_variable('bias', shape=[C_out],
            dtype=dtype, initializer=tf.zeros_initializer, trainable=trainable),
            [1,1,C_out])

        if mask is not None:
            mask = tf.tile(tf.reduce_min(mask, axis=-1, keepdims=True), [1,1,C_out])
            mask = mask[:, cutter[0]:shape(mask)[1]-cutter[1], :]
            tensor *= mask

        return tensor, mask



##### TODO fix down #####
def modulated_conv2d(
        tensor,
        channels,
        kernel_size,
        style=None,
        mask=None,
        demodulate=True,
        fused=True,
        padding='SAME',
        input_gain=None,
        epsilon=1e-8,
        bias=True,
        trainable=True,
        scope='mod_conv2d'):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, _, _, C_in = shape(tensor)
        [kH, kW], C_out = kernel_size, channels
        padding=padding.upper()
        if padding == 'SAME':
            tensor = tf.pad(tensor, [[0,0], [(kH-1)//2,kH//2], [(kW-1)//2,kW//2], [0,0]])
            padding = 'VALID'

        weight = tf.get_variable('weight', shape=[kH, kW, C_in, C_out], dtype=dtype,
                initializer=tf.random_normal_initializer, trainable=trainable)
        ww = weight[None]

        if style is not None:
            style = linear(style, C_in, trainable=trainable, scope='linear')[0] + 1.
            ww *= tf.cast(style[:, None, None, :, None], dtype)

        if demodulate:
            demod = tf.math.rsqrt(tf.maximum(tf.einsum('bhwio,bhwio->bo', ww, ww), epsilon))
            ww *= demod[:, None, None, None]

        if input_gain is not None:
            ww *= tf.broadcast_to(input_gain, [shape(ww)[0],1,1,C_in,1])

        if style is None:
            ww = tf.squeeze(ww, axis=0) # hwio
        elif fused:
            t_shape = [1,*shape(tensor)[1:3],B*C_in]
            tensor = tf.reshape(tf.transpose(tensor, [1,2,0,3]), t_shape)
            ww = tf.reshape(tf.transpose(ww, [1,2,3,0,4]), [kH,kW,C_in,B*C_out])
        else:
            tensor *= tf.cast(style[:, None, None, :], dtype=dtype) # bhwi

        tensor = tf.nn.conv2d(tensor, ww, strides=[1,1,1,1], padding=padding)

        if fused and style is not None:     # tensor : 1, H, W, B*C_out
            tensor = tf.transpose(tf.reshape(tensor, [*shape(tensor)[1:-1],B,C_out]),
                    [2,0,1,3])
        elif demodulate:
            tensor *= tf.cast(demod[:, None, None], dtype=dtype)

        if bias: tensor += tf.broadcast_to(tf.get_variable('bias', shape=[C_out],
            dtype=dtype, initializer=tf.zeros_initializer, trainable=trainable),
            [1,1,1,C_out])

        if mask is not None:
            mask = tf.tile(tf.reduce_min(mask, axis=-1, keepdims=True), [1,1,1,C_out])
            tensor *= mask

        return tensor, mask



##### TODO fix down #####
def alias_free_resample_1d(
        tensor,
        channels,
        kernel_size=1,
        upsampling_factor=1,
        downsampling_factor=1,
        mask=None,
        filter_size=6,
        #use_radial_filters=False,
        #critically_sampled=False,
        trainable=True,
        scope='af_re_1d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, L, C_in = shape(tensor)
        if mask is None:
            smp_rate = tf.ones([B * L], dtype=dtype)
        else:
            mask_bl = tf.reduce_max(mask, axis=-1)
            smp_rate = tf.reduce_sum(mask_bl, axis=-1)
        up_filter = FIR_lpf(smp_rate, filter_size, up_factor=upsampling_factor)
        down_filter = FIR_lpf(smp_rate, filter_size, up_factor=upsampling_factor,
                down_factor=downsampling_factor)

        top_size = L * upsampling_factor
        out_size = top_size // downsampling_factor
        #tap_size = filter_size*(upsampling_factor + downsampling_factor) - 2
        uptaps = upsampling_factor * filter_size
        up_taps = max(0, (int(upsampling_factor > 1) * uptaps) - 1)
        downtaps = downsampling_factor * filter_size
        down_taps = max(0, (int(downsampling_factor > 1) * downtaps) - 1)
        tap_size = up_taps + down_taps
        pad_total = ((out_size-1) * downsampling_factor)+1
        pad_total -= top_size
        pad_total += tap_size
        pad_total = tf.cast(pad_total, dtype=tf.int32)
        pad_left = (pad_total + upsampling_factor)//2
        pad_right = pad_total - pad_left

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_upfir_pl', pad_left),
        #    tf_print(f'{scope}_upfir_pr', pad_right),
        #    ]):
        #    tensor = tf.identity(tensor)
        #with tf.control_dependencies([
        #    tf_print('tensor', tensor, color='cyan'),
        #    tf_print('mask', mask, color='cyan')
        #    ]):
        #    tensor = tf.identity(tensor)

        tensor, mask = upfirdn_1d(tensor,
                kernel=up_filter if upsampling_factor > 1 else None,
                u=upsampling_factor, pl=pad_left, pr=pad_right,
                gain=upsampling_factor**2, mask=mask)

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_upfir', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        if C_in != channels:
            weight = tf.get_variable('weight', [kernel_size, C_in, channels], dtype=dtype,
                    initializer=kaiming_uniform_init(kernel_size*C_in), trainable=trainable)
            tensor = tf.nn.conv1d(tensor, weight, stride=[1,1,1], padding='SAME')
            if mask is not None:
                mask = tf.tile(tf.reduce_max(mask, axis=-1, keepdims=True), [1,1,channels])
                tensor *= mask

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_conv_out', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        tensor, mask = upfirdn_1d(tensor,
                kernel=down_filter if downsampling_factor > 1 else None,
                d=downsampling_factor, mask=mask)

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_downfir', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        return tensor, mask


##### TODO fix down #####
def alias_free_mux_1d(
        tensor,
        channels,
        latent=None,
        kernel_size=1,
        upsampling_factor=1,
        downsampling_factor=1,
        mask=None,
        activation=gelu,
        activation_gain=np.sqrt(2.),
        filter_size=6,
        #use_radial_filters=False,
        #critically_sampled=False,
        update_emas=False,
        magnitude_ema_beta=0.999,
        trainable=True,
        scope='af_mux_1d'
        ):
    with tf.variable_scope(scope):
        dtype = tensor.dtype
        B, L, C_in = shape(tensor)
        if mask is None:
            smp_rate = 2*tf.ones([B*L], dtype=dtype)
        else:
            mask_bl = tf.reduce_max(mask, axis=-1)
            smp_rate = 2*tf.reduce_sum(mask_bl, axis=-1)

        if update_emas:
            mag_ema = tf.get_variable('magnitude_ema', shape=[], dtype=dtype,
                    initializer=tf.ones_initializer, trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            mag_cur = tf.einsum('blc,blc->b', tensor, tensor)
            mag_div = tf.einsum('blc->b', mask) if mask is not None\
                    else tf.tile([L*C_in], [B])
            mag_div = tf.cast(mag_div, dtype=dtype)
            mag_cur = tf.reduce_mean(mag_cur / tf.where(tf.equal(mag_div, 0.0),
                1e-8 * tf.ones([B], dtype=dtype), mag_div))
            mag_new = mag_cur + magnitude_ema_beta * (mag_ema - mag_cur)
            with tf.control_dependencies([tf.assign(mag_ema, mag_new)]):
                input_gain = tf.math.rsqrt(mag_ema)

        up_filter = FIR_lpf(smp_rate, filter_size, up_factor=upsampling_factor)
        down_filter = FIR_lpf(smp_rate, filter_size, up_factor=upsampling_factor,
                down_factor=downsampling_factor)

        top_size = L * upsampling_factor
        out_size = top_size // downsampling_factor
        #tap_size = filter_size*(upsampling_factor + downsampling_factor) - 2
        uptaps = upsampling_factor * filter_size
        up_taps = max(0, (int(upsampling_factor > 1) * uptaps) - 1)
        downtaps = downsampling_factor * filter_size
        down_taps = max(0, (int(downsampling_factor > 1) * downtaps) - 1)
        tap_size = up_taps + down_taps
        pad_total = ((out_size-1) * downsampling_factor)+1
        pad_total -= top_size
        pad_total += tap_size
        pad_total = tf.cast(pad_total, dtype=tf.int32)
        pad_left = (pad_total + upsampling_factor)//2
        pad_right = pad_total - pad_left

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_upfir_pl', pad_left),
        #    tf_print(f'{scope}_upfir_pr', pad_right),
        #    tf_print('tensor', tensor),
        #    tf_print('mask', mask),
        #    ]):
        #    tensor = tf.identity(tensor)

        tensor, mask = upfirdn_1d(tensor,
                kernel=up_filter if upsampling_factor > 1 else None,
                u=upsampling_factor, pl=pad_left, pr=pad_right,
                gain=upsampling_factor**2, mask=mask)

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_upfir', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        tensor, mask = modulated_conv1d(tensor, channels, kernel_size, style=latent,
                mask=mask, demodulate=True, padding='SAME',
                input_gain=input_gain if update_emas else 1., bias=True,
                trainable=trainable, scope='mod_conv1d')

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_conv_out', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        if activation is not None:
            tensor = activation(tensor)
            if activation_gain != 1: tensor *= tf.cast(activation_gain, dtype)

        tensor, mask = upfirdn_1d(tensor,
                kernel=down_filter if downsampling_factor > 1 else None,
                d=downsampling_factor, mask=mask)

        #with tf.control_dependencies([
        #    tf_print(f'{scope}_downfir', tensor),
        #    ]):
        #    tensor = tf.identity(tensor)

        return tensor, mask


##### TODO fix down #####
def alias_free_resample_2d(
        tensor,
        channels,
        kernel_size=(1,1),
        upsampling_factors=(1,1),
        downsampling_factors=(1,1),
        mask=None,
        filter_size=6,
        #use_radial_filters=False,
        #critically_sampled=False,
        trainable=True,
        scope='af_re_2d'
        ):
    with tf.variable_scope(scope):
        ### CAREFUL : stopband = cutoff + half_width <= 0.5 * sampling_rate
        # setting in Alias-Free GAN : 0.25 * stopband <= cutoff <= 0.5 * stopband
        dtype = tensor.dtype
        B, H, W, C_in = shape(tensor)
        if mask is None:
            #smp_rates = [tf.ones([B, SR], dtype=dtype) for SR in [H, W]]
            smp_rates = list_elemops([tf.ones([B], dtype=dtype)]*2, [H, W], mode='*')
        else:
            mask_bhw = tf.reduce_max(mask, axis=-1) # BHW
            smp_rates = [tf.reduce_sum(mask_bhw, axis=N_D)[:,0] for N_D in [1, 2]]

        upH_filter = FIR_lpf(smp_rates[0], filter_size,
                up_factor=upsampling_factors[0])
        upW_filter = FIR_lpf(smp_rates[1], filter_size,
                up_factor=upsampling_factors[1])
        downH_filter = FIR_lpf(smp_rates[0], filter_size,
                up_factor=upsampling_factors[0], down_factor=downsampling_factors[0])
        downW_filter = FIR_lpf(smp_rates[1], filter_size,
                up_factor=upsampling_factors[1], down_factor=downsampling_factors[1])

        # padding
        top_rates = list_elemops([H,W], upsampling_factors, mode='*')
        output_size = list_elemops(top_rates, downsampling_factors, mode='//')
        uptaps_H, uptaps_W = list_elemops(upsampling_factors, [filter_size]*2, mode='*')
        up_taps = [int(upsampling_factors[0] > 1) * uptaps_H,
                int(upsampling_factors[1] > 1) * uptaps_W]
        up_taps = [max(0, u-1) for u in up_taps]
        downtaps_H, downtaps_W = list_elemops(downsampling_factors, [filter_size]*2, mode='*')
        down_taps = [int(downsampling_factors[0] > 1) * downtaps_H,
                int(downsampling_factors[1] > 1) * downtaps_W]
        down_taps = [max(0, d-1) for d in down_taps]
        tap_sizes = list_elemops(up_taps, down_taps, mode='+')
        pad_total = list_elemops_iter((output_size, [1,1], downsampling_factors, [1,1],
            top_rates, tap_sizes), modes=('-', '*', '+', '-', '+'))
        pad_total = list_elemops(pad_total, mode=partial(tf.cast, dtype=tf.int32))
        pad_left = list_elemops_iter((pad_total, upsampling_factors, [2,2]), modes=('+', '//'))
        pad_left = list_elemops(pad_left, mode=partial(tf.cast, dtype=tf.int32))
        pad_right = list_elemops(pad_total, pad_left, mode='-')
        pad_right = list_elemops(pad_right, mode=partial(tf.cast, dtype=tf.int32))

        #if np.prod(upsampling_factors) > 1:
        # apply filter
        tensor, mask = upfirdn_2d(tensor,
                kernel_h=upH_filter if upsampling_factors[0] > 1 else None,
                kernel_w=upW_filter if upsampling_factors[1] > 1 else None,
                uh=upsampling_factors[0], uw=upsampling_factors[1], phl=pad_left[0],
                phr=pad_right[0], pwl=pad_left[1], pwr=pad_right[1],
                gain_h=upsampling_factors[0] ** 2, gain_w=upsampling_factors[1] ** 2,
                mask=mask)

        #if np.prod(downsampling_factors) > 1:
        # apply filter
        tensor, mask = upfirdn_2d(tensor,
                kernel_h=downH_filter if downsampling_factors[0] > 1 else None,
                kernel_w=downW_filter if downsampling_factors[1] > 1 else None,
                dh=downsampling_factors[0], dw=downsampling_factors[1], mask=mask)

        if C_in != channels:
            weight = tf.get_variable('weight', shape=[*kernel_size, C_in, channels], dtype=dtype,
                    initializer=tf.glorot_uniform_initializer, trainable=trainable)
            tensor = tf.nn.conv2d(tensor, weight, strides=[1,1,1,1], padding='SAME')
            mask = tf.tile(tf.reduce_max(mask, axis=-1, keepdims=True), [1,1,1,channels])
            tensor *= mask

        return tensor, mask


##### TODO fix down #####
def alias_free_mux_2d(
        tensor,
        channels,
        latent=None,
        kernel_size=(3,3),
        upsampling_factors=(1, 1),
        downsampling_factors=(1, 1),
        mask=None,
        activation=gelu,
        activation_gain=np.sqrt(2.),
        filter_size=6,
        #use_radial_filters=False,
        #critically_sampled=False,
        update_emas=False,
        magnitude_ema_beta=0.999,
        trainable=True,
        scope='af_mux_2d'
        ):
    with tf.variable_scope(scope):
        ### CAREFUL : stopband = cutoff + half_width <= 0.5 * sampling_rate
        # setting in Alias-Free GAN : 0.25 * stopband <= cutoff <= 0.5 * stopband
        dtype = tensor.dtype
        B, H, W, C_in = shape(tensor)
        if mask is None:
            #smp_rates = [tf.ones([B, SR], dtype=dtype) for SR in [H, W]]
            smp_rates = [tf.tile([SR], [B]) for SR in [H, W]]
        else:
            mask_bhw = tf.reduce_max(mask, axis=-1) # BHW
            smp_rates = [tf.reduce_sum(mask_bhw, axis=N_D)[:, 0] for N_D in [1, 2]]

        if update_emas:
            mag_ema = tf.get_variable('magnitude_ema', shape=[], dtype=dtype,
                    initializer=tf.ones_initializer, trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, "MOVING_AVERAGE"])
            mag_cur = tf.einsum('bwhc,bwhc->b', tensor, tensor)
            mag_div = tf.einsum('bwhc->b', mask) if mask is not None\
                    else tf.tile([H*W*C_in], [B])
            mag_div = tf.cast(mag_div, dtype=dtype)
            mag_cur = tf.reduce_mean(mag_cur / tf.where(tf.equal(mag_div, 0.0),
                1e-8 * tf.ones([B], dtype=dtype), mag_div))
            mag_new = mag_cur + magnitude_ema_beta * (mag_ema - mag_cur)
            with tf.control_dependencies([tf.assign(mag_ema, mag_new)]):
                input_gain = tf.math.rsqrt(tf.maximum(mag_ema, 1e-8))

        #modulated_conv2d
        tensor, mask = modulated_conv2d(tensor, channels, kernel_size, style=latent,
                mask=mask, demodulate=True, padding='SAME',
                input_gain=input_gain if update_emas else 1., bias=True,
                trainable=trainable, scope='mod_conv2d')

        upH_filter = FIR_lpf(smp_rates[0], filter_size,
                up_factor=upsampling_factors[0])
        upW_filter = FIR_lpf(smp_rates[1], filter_size,
                up_factor=upsampling_factors[1])
        downH_filter = FIR_lpf(smp_rates[0], filter_size,
                up_factor=upsampling_factors[0], down_factor=downsampling_factors[0])
        downW_filter = FIR_lpf(smp_rates[1], filter_size,
                up_factor=upsampling_factors[1], down_factor=downsampling_factors[1])

        # padding
        top_rates = list_elemops([H,W], upsampling_factors, mode='*')
        output_size = list_elemops(top_rates, downsampling_factors, mode='//')
        uptaps_H, uptaps_W = list_elemops(upsampling_factors, [filter_size]*2, mode='*')
        up_taps = [int(upsampling_factors[0] > 1) * uptaps_H,
                int(upsampling_factors[1] > 1) * uptaps_W]
        up_taps = [max(0, u-1) for u in up_taps]
        downtaps_H, downtaps_W = list_elemops(downsampling_factors, [filter_size]*2, mode='*')
        down_taps = [int(downsampling_factors[0] > 1) * downtaps_H,
                int(downsampling_factors[1] > 1) * downtaps_W]
        down_taps = [max(0, d-1) for d in down_taps]
        tap_sizes = list_elemops(up_taps, down_taps, mode='+')
        #up_taps = list_elemops(upsampling_factors, [filter_size]*2, mode='*')
        #down_taps = list_elemops(downsampling_factors, [filter_size]*2, mode='*')
        #tap_sizes = list_elemops_iter((up_taps, down_taps, [2,2]), modes=('+', '-'))
        pad_total = list_elemops_iter((output_size, [1,1], downsampling_factors, [1,1],
            top_rates, tap_sizes), modes=('-', '*', '+', '-', '+'))
        pad_total = list_elemops(pad_total, mode=partial(tf.cast, dtype=tf.int32))
        pad_left = list_elemops_iter((pad_total, upsampling_factors, [2,2]), modes=('+', '//'))
        pad_left = list_elemops(pad_left, mode=partial(tf.cast, dtype=tf.int32))
        pad_right = list_elemops(pad_total, pad_left, mode='-')
        pad_right = list_elemops(pad_right, mode=partial(tf.cast, dtype=tf.int32))

        #if np.prod(upsampling_factors) > 1:
        # apply filter
        tensor, mask = upfirdn_2d(tensor,
                kernel_h=upH_filter if upsampling_factors[0] > 1 else None,
                kernel_w=upW_filter if upsampling_factors[1] > 1 else None,
                uh=upsampling_factors[0], uw=upsampling_factors[1], phl=pad_left[0],
                phr=pad_right[0], pwl=pad_left[1], pwr=pad_right[1],
                gain_h=upsampling_factors[0] ** 2, gain_w=upsampling_factors[1] ** 2,
                mask=mask)

        # activation
        if activation is not None:
            tensor = activation(tensor)
            if activation_gain != 1: tensor *= tf.cast(activation_gain, tensor.dtype)
            if mask is not None: tensor *= mask

        #if np.prod(downsampling_factors) > 1:
        # apply filter
        tensor, mask = upfirdn_2d(tensor,
                kernel_h=downH_filter if downsampling_factors[0] > 1 else None,
                kernel_w=downW_filter if downsampling_factors[1] > 1 else None,
                dh=downsampling_factors[0], dw=downsampling_factors[1],
                mask=mask)

        return tensor, mask


