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



