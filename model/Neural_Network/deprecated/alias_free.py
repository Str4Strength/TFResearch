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


