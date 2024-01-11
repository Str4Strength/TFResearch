import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .Neural_Networks import *
from .utils import *




R_Q = 0.1
R_D = 0.1



def res_scale(value, dtype, trainable=True, scope='res_scale'):
    scale = tf.get_variable(scope, [], dtype=dtype, initializer=tf.constant_initializer(value=value),
            trainable=trainable)
    scale = tf.minimum(tf.math.abs(scale), 1)
    return scale, tf.sqrt(1 - (scale ** 2))



def shortcut(
        tensor,
        features,
        mask=None,
        blocks=8,
        trainable=True,
        scope='shortcut'
        ):
    with tf.variable_scope('skip'):

        if features == shape(tensor)[-1]:
            sc_out, sc_mask = tensor, mask
        else:
            sc_out, sc_mask = linear(tensor, out_features=int(features), mask=mask, quantization=R_Q,
                    quantization_blocks=blocks, trainable=trainable, scope='proj')

        #sc_out = tf.where(tf.equal(sc_mask, 0), tf.zeros_like(sc_out), tf.nn.softmax(sc_out, axis=-1))

        return sc_out, sc_mask



def dynamic_latent_conv(
        tensor,
        latent,
        features,
        kernel_size=5,
        mask=None,
        quantization=0,
        quantization_blocks=0,
        trainable=True,
        scope='dynamic_latent'
        ):
    with tf.variable_scope(scope):
        t_shape, l_shape = shape(tensor), shape(latent)
        latent = tf.reshape(latent, (t_shape[0], t_shape[-1], l_shape[-1] // t_shape[-1]))

        weight = linear(latent, out_features=(kernel_size, features), quantization=R_Q * 0.1,
                quantization_blocks=l_shape[-1] // t_shape[-1], trainable=trainable, scope='weight')[0]   # b, i, k, o
        weight *= tf.math.rsqrt(tf.maximum(tf.reduce_sum(weight ** 2., axis=(1, 2), keepdims=True), 1e-8))

        tensor = tf.reshape(tf.transpose(tensor, [1, 0, 2]), (1, t_shape[1], t_shape[0] * t_shape[2]))
        weight = tf.reshape(tf.transpose(weight, [2, 1, 0, 3]), (kernel_size, t_shape[-1], t_shape[0] * features))
        weight = quantization_noise(weight, in_features=(t_shape[-1],), out_features=shape(weight)[-1],
                p=quantization, block_size=quantization_blocks)
        tensor = tf.nn.conv1d(tensor, weight, stride=1, padding='SAME')
        tensor = tf.transpose(tf.reshape(tensor, [t_shape[1], t_shape[0], features]), [1, 0, 2])

        tensor += linear(latent, in_features=shape(latent)[1:], out_features=features, quantization=R_Q, quantization_blocks=np.prod(shape(latent)[1:]), trainable=trainable, scope='bias')[0][:, None]

        mask = reconstruct_mask(features, mask)
        if exists(mask): tensor *= mask

        return tensor, mask



def concat_latent(
        tensor,
        latent,
        mask=None,
        trainable=True,
        scope='concat_latent'
        ):
    with tf.variable_scope(scope):

        features = shape(tensor)[-1]
        _, lat_features = shape(latent)

        latent = linear(latent, out_features=features, quantization=R_Q, quantization_blocks=lat_features,  trainable=trainable, scope='linear_in')[0]
        latent = mish(latent)
        latent = linear(latent, out_features=features // 4, trainable=trainable, scope='linear_out')[0]

        latent = tf.tile(latent[:, None], [1, shape(tensor)[1], 1])

        return tf.concat([tensor, latent], axis=-1), reconstruct_mask(5 * features // 4, mask)



def concat_sequence(
        tensor,
        sequence,
        mask=None,
        trainable=True,
        scope='concat_sequence'
        ):
    with tf.variable_scope(scope):

        #features = shape(tensor)[-1]

        #sequence = feed_forward(sequence, features=features // 8, mask=reconstruct_mask(shape(sequence)[-1], mask),
        #        activation=mish, quantization=R_Q, quantization_blocks=4, trainable=trainable, scope='ff')[0]
        #sequence = deconvolution(sequence, 2, int(features // 4), kernels=3, strides=(1, int(shape(tensor)[-2] // shape(sequence)[-2])),
        #        mask=reconstruct_mask(shape(sequence)[-1], mask)[:, :, :shape(sequence)[-2]],
        #        trainable=trainable, scope='deconv')
        #with tf.control_dependencies([tf_print('tensor', tensor), tf_print('pitch', sequence)]): tensor = tf.identity(tensor)
        tensor = tf.concat([sequence, tensor], axis=-1)
        mask = reconstruct_mask(shape(tensor)[-1], mask)
        if exists(mask): tensor *= mask

        return tensor, mask


def add_sequence(
        tensor,
        sequence,
        activation=mish,
        mask=None,
        trainable=True,
        scope='add_sequence'):
    with tf.variable_scope(scope):

        tensor = normalization(tensor, group_size=16, mask=mask, trainable=trainable, scope='group_norm')

        sequence, mask_seq = linear(sequence, out_features=2 * shape(tensor)[-1], mask=reconstruct_mask(shape(sequence)[-1], mask),
                quantization=R_Q, quantization_blocks=shape(sequence)[-1], trainable=trainable, scope='seq_linear')
        sequence = activation(sequence) * mask_seq if exists(mask) else activation(sequence)
        sequence, _ = convolution(sequence, 1, shape(tensor)[-1], 3, mask=mask_seq, trainable=trainable, scope='seq_conv1d')

        rt, rs = res_scale(1.0, tensor.dtype, trainable=trainable, scope='add_ratio')
        tensor = (rt * tensor) + (rs * sequence)

        return tensor, mask


# TODO if needed
def fusion(
        tensor,
        sequence,
        mask=None,
        mask_seq=None,
        trainable=True,
        scope='fusion'
        ):
    with tf.variable_scope(scope):
        heads = shape(tensor)[-1] // 64

        tensor = normalization(tensor, heads, mask=mask, trainable=trainable, scope='norm_tensor')
        sequence = normalization(sequence, 16, mask=mask_seq, trainable=trainable, scope='norm_seq')

        res_out, _, _, _ = attention(tensor, sequence, sequence, out_features=shape(tensor)[-1], heads=heads, mask_query=mask,
                mask_key=mask_seq, mask_value=mask_seq, quantization=R_Q, quantization_blocks=16, trainable=trainable,
                scope='attn')

        #res_out, _, _ = fast_attention(tensor, sequence, sequence, out_features=shape(tensor)[-1], heads=heads, mask_query=mask,
        #        mask_key=mask_seq, mask_value=mask_seq, quantization=R_Q, quantization_blocks=16, trainable=trainable,
        #        scope='fast_attn')

        #res_out, _, _ = proximal_attention(tensor, sequence, sequence, window_size=7, out_features=shape(tensor)[-1], heads=heads,
        #        mask_query=mask, mask_key=mask_seq, mask_value=mask_seq, quantization=R_Q, quantization_blocks=16, trainable=trainable,
        #        scope='prox_attn')

        #res_out, _, _ = hybrid_fast_attention(tensor, sequence, sequence, window_size=7, out_features=shape(tensor)[-1],
        #        full_heads=heads // 2, prox_heads=heads // 2, mask_query=mask, mask_key=mask_seq, mask_value=mask_seq,
        #        quantization=R_Q, quantization_blocks=16, trainable=trainable, scope='hybrid_attn')

        re_scale, sc_scale = res_scale(0, tensor.dtype, trainable=trainable, scope='res_scale')
        tensor = re_scale * res_out + sc_scale * tensor

        return tensor, mask



def res_blk(
        tensor,
        features,
        activation=mish,
        mask=None,
        trainable=True,
        scope='res_blk',
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #TODO 7 reduce, 4 * features 를 줄이기
            res_out, res_mask = convolution(res_out, 1, 2 * features, 5, mask=res_mask, quantization=R_Q, quantization_blocks=5, trainable=trainable, scope='conv1d_0')
            res_out *= (5 ** -0.5)
            res_out = activation(res_out) * res_mask if exists(mask) else activation(res_out)
            if trainable: res_out = tf.nn.dropout(res_out, rate=R_D, name='drop')
            res_out, res_mask = convolution(res_out, 1, features, 3, mask=res_mask, trainable=trainable, scope='conv1d_1')
            res_out *= (3 ** -0.5)

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        #tensor = (res_out + sc_out) * (2 ** -0.5)
        rt, rs = res_scale(2 ** -0.5, tensor.dtype, trainable=trainable, scope='res_scalar')
        tensor = (rt * res_out) + (rs * sc_out)
        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask



def sty_blk(
        tensor,
        latent,
        features,
        activation=mish,
        mask=None,
        trainable=True,
        scope='sty_blk',
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')

            res_out, res_mask = dynamic_latent_conv(res_out, latent, 4 * features, 9, mask=res_mask, quantization=R_Q, quantization_blocks=9, trainable=trainable, scope='conv1d_0')
            res_out *= (9 ** -0.5)
            res_out = activation(res_out) * res_mask if exists(mask) else activation(res_out)
            if trainable: res_out = tf.nn.dropout(res_out, rate=R_D, name='drop')
            res_out, res_mask = convolution(res_out, 1, features, 3, mask=res_mask, trainable=trainable, scope='conv1d_1')

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        tensor = (res_out + sc_out) * (2 ** -0.5)
        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask



def sattn_blk(
        tensor,
        features,
        mask=None,
        trainable=True,
        scope='attn_blk'
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #cprint(f'{scope}_{features}_{shape(res_out)[-1]}', color='blue')
            res_out, res_mask, _, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
                    quantization=R_Q, quantization_blocks=shape(res_out)[-1] // 4,
                    trainable=trainable, scope='attn')

            #res_out, res_mask, _ = fast_attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=features // 4,
            #        trainable=trainable, scope='fast_attn')

            #res_out, res_mask, _ = proximal_attention(tensor, res_out, res_out, window_size=7, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask, quantization=R_Q, quantization_blocks=16, trainable=trainable,
            #        scope='prox_attn')

            #res_out, res_mask, _ = hybrid_fast_attention(tensor, res_out, res_out, window_size=7, out_features=features,
            #        full_heads=features // 128, prox_heads=features // 128, mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=16, trainable=trainable, scope='hybrid_attn')

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        re_scale, sc_scale = res_scale(0, tensor.dtype, trainable=trainable, scope='res_scale')
        tensor = re_scale * res_out + sc_out #sc_scale * sc_out
        if exists(mask): tensor *= sc_mask

        return tensor, sc_mask, res_out


# TODO 여기부터
def cattn_blk(
        tensor,
        style
        features,
        mask=None,
        trainable=True,
        scope='attn_blk'
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #cprint(f'{scope}_{features}_{shape(res_out)[-1]}', color='blue')
            res_out, res_mask, _, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
                    quantization=R_Q, quantization_blocks=shape(res_out)[-1] // 4,
                    trainable=trainable, scope='attn')

            #res_out, res_mask, _ = fast_attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=features // 4,
            #        trainable=trainable, scope='fast_attn')

            #res_out, res_mask, _ = proximal_attention(tensor, res_out, res_out, window_size=7, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask, quantization=R_Q, quantization_blocks=16, trainable=trainable,
            #        scope='prox_attn')

            #res_out, res_mask, _ = hybrid_fast_attention(tensor, res_out, res_out, window_size=7, out_features=features,
            #        full_heads=features // 128, prox_heads=features // 128, mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=16, trainable=trainable, scope='hybrid_attn')

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        re_scale, sc_scale = res_scale(0, tensor.dtype, trainable=trainable, scope='res_scale')
        tensor = re_scale * res_out + sc_out #sc_scale * sc_out
        if exists(mask): tensor *= sc_mask

        return tensor, sc_mask, res_out



def fattn_blk(
        tensor,
        features,
        mask=None,
        trainable=True,
        scope='attn_blk'
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #cprint(f'{scope}_{features}_{shape(res_out)[-1]}', color='blue')
            #res_out, res_mask, _, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=shape(res_out)[-1] // 4,
            #        trainable=trainable, scope='attn')

            res_out, res_mask, _ = fast_attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
                    quantization=R_Q, quantization_blocks=features // 4,
                    trainable=trainable, scope='fast_attn')

            #res_out, res_mask, _ = proximal_attention(tensor, res_out, res_out, window_size=7, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask, quantization=R_Q, quantization_blocks=16, trainable=trainable,
            #        scope='prox_attn')

            #res_out, res_mask, _ = hybrid_fast_attention(tensor, res_out, res_out, window_size=7, out_features=features,
            #        full_heads=features // 128, prox_heads=features // 128, mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=16, trainable=trainable, scope='hybrid_attn')

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        re_scale, sc_scale = res_scale(0, tensor.dtype, trainable=trainable, scope='res_scale')
        tensor = re_scale * res_out + sc_out #sc_scale * sc_out
        if exists(mask): tensor *= sc_mask

        return tensor, sc_mask, res_out



def upsample_double(
        tensor,
        mask=None,
        ):
    b, t, f = shape(tensor)
    expander = tf.stack([tensor, tf.pad(tensor[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'SYMMETRIC')], axis=-2)
    expander = tf.reduce_mean(expander, axis=-2)
    tensor = tf.stack([tensor, expander], axis=-2)
    tensor = tf.reshape(tensor, [b, 2* t, f])

    if exists(mask):
        mask = tf.reshape(tf.tile(mask[:, :, None], [1, 1, 2, 1]), [b, 2*t, f])
        tensor *= mask

    return tensor, mask



def encoder_block(
        tensor,
        features,
        mask=None,
        trainable=True,
        scope='enc_blk'
        ):
    with tf.variable_scope(scope):

        tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')
        tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res')
        #with tf.control_dependencies([tf_print(f'{scope}_res', tensor, color='green')]): tensor = tf.identity(tensor)
        #tensor, mask = res_blk(tensor, features, group_size=16, activation=mish, mask=mask, trainable=trainable, scope='res_1')

        return tensor, mask



def decoder_block(
        tensor,
        features,
        reference,
        sequence=None,
        mask=None,
        mask_ref=None,
        trainable=True,
        scope='dec_blk'
        ):
    with tf.variable_scope(scope):

        #tensor, mask = concat_sequence(tensor, sequence, mask=mask, trainable=trainable, scope='cat_seq')

        #tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res')

        ### option 1
        #tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')
        #tensor, mask = sty_blk(tensor, latent, features, activation=mish, mask=mask, trainable=trainable, scope='sty')

        ### option 2
        #tensor, mask = concat_latent(tensor, latent, mask=mask, trainable=trainable, scope='cat_lat')
        if exists(sequence): tensor, mask = add_sequence(tensor, sequence, activation=mish, mask=mask, trainable=trainable, scope='add_seq')
        tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')

        """
        reference = normalization(reference, group_size=32, mask=mask_ref, trainable=trainable, scope='ref_group_norm')
        reference, mask_ref = convolution(reference, 1, shape(tensor)[-1], 3, mask=mask_ref, trainable=trainable, scope='ref_conv1d')

        tensor = tf.concat([reference, tensor], axis=1)
        if exists(mask):
            if not exists(mask_ref): mask_ref = tf.ones_like(reference)
            mask_ref = tf.concat([mask_ref, mask], axis=1)
        else:
            if exists(mask_ref): mask_ref = tf.concat([mask_ref, tf.ones_like(tensor)])
        tensor, _, _ = attn_blk(tensor, features, mask=mask_ref, trainable=trainable, scope='attn_sty')
        tensor = tensor[:, shape(reference)[1]:]
        """

        tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res_ff')

        ### option 3
        #cat_lat = linear(latent, out_features=shape(latent)[-1], trainable=trainable, scope='lat_type_cat')[0]
        #dyn_lat = linear(latent, out_features=shape(latent)[-1], trainable=trainable, scope='lat_type_dyn')[0]

        #tensor, mask = concat_latent(tensor, latent, mask=mask, trainable=trainable, scope='cat_lat')
        #tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')
        #tensor, mask = sty_blk(tensor, latent, features, activation=mish, mask=mask, trainable=trainable, scope='sty')


        return tensor, mask


def upsize_block(
        tensor,
        features,
        #sequence,
        latent=None,
        mask=None,
        trainable=True,
        scope='dec_blk'
        ):
    with tf.variable_scope(scope):
        b, l, d = shape(tensor)
        tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')
        tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res_ff')
        return tensor, mask


def squeeze_sequence(
        seq,
        rate,
        ):
    n_tok = shape(seq)[-2]
    p_f, p_b = 0, -n_tok % 2
    p_f, p_b = p_f + rate - 1, p_b + rate - 1
    front = tf.reduce_mean(seq[:, :2 * rate - p_f], axis=1, keepdims=True)
    sqz_seq = tf.nn.avg_pool1d(seq[:, rate - p_f : n_tok - rate + p_b], 2 * rate, rate, padding='VALID')
    back = tf.reduce_mean(seq[:, - 2 * rate + p_b:], axis=1, keepdims=True)
    sqz_seq = tf.concat([front, sqz_seq, back], axis=1)

    return sqz_seq



def network(
        tensor,
        reference,
        pitch,
        #translate_weights,
        features = 128,
        kernel = 8,
        stride = 4,
        size_rates = [2, 2, 2],
        encodings = 3,
        decodings = 5,
        fusion_maps = None,
        fusion_masks = None,
        min_values=0.0,
        mask = None,
        mask_ref = None,
        train = True,
        scope = 'gen_net',
        reuse=tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):
        assert kernel > stride and kernel % stride == 0
        # size_rates - s_0 ~ s_n-1   N = n
        # cump_rates - 1, s_0, ~ , s_0...s_n-1   N = n+1
        # units - u, u * s_0, ~ , u * s_0...s_n-1   N = n+1
        # masks - mask, mask_/s_0, ~ , mask_/s_0...s_n-1   N = n+1
        size = shape(tensor)
        #pitches = [pitch]
        #for n in size_rates: pitches.append(squeeze_sequence(pitches[-1], n))

        with tf.variable_scope('initial'):
            #kernel, stride = 6, 3 #TODO h24k_v2_sing
            #enc_features, dec_features = features, features #TODO h24k_v2_sing
            #kernel, stride = 32, 16 #TODO h24k_16_sing
            #kernel, stride = 64, 32 #TODO h24k_32_sing
            ratio = int(np.floor(np.sqrt(stride)))
            enc_features, dec_features = int(features * ratio), features #TODO h24k_16_sing, further

            """
            P_in = -size[-2]%stride
            P_L, P_R = (kernel - 1) // 2, kernel // 2
            x = tf.pad(tensor, ((0, 0), (P_L, P_R + P_in), (0, 0)), mode='CONSTANT', constant_values=min_values)
            mask_x = tf.pad(mask, ((0, 0), (P_L, P_R + P_in), (0, 0)), mode='CONSTANT', constant_values=0.0) if exists(mask) else None

            x, mask_x = convolution(x, 1, enc_features, kernel, stride, padding='valid', mask=mask_x, trainable=train, scope='proj_conv')
            """
            # init proj
            x, mask_x = linear(tensor, out_features=enc_features, mask=mask, trainable=train, scope='proj_lin')
            #with tf.control_dependencies([tf_print('proj_lin', x, color='magenta')]): x = tf.identity(x)

            # cut & downsample
            x = normalization(x, group_size=16, mask=mask_x, trainable=train, scope='group_norm')
            #with tf.control_dependencies([tf_print('proj_norm', x, color='magenta')]): x = tf.identity(x)
            x, mask_x = convolution(x, 1, enc_features, kernel, stride, padding='same', mask=mask_x, trainable=train, scope='proj_conv')

            #with tf.control_dependencies([tf_print('proj_out', x, color='magenta')]): x = tf.identity(x)

        # if translate_weights: TODO implement

        with tf.variable_scope('intermediate'):
            fmaps, mask_n = [], mask_x

            for n in range(encodings): x, mask_x = encoder_block(x, enc_features, mask=mask_x, trainable=train, scope=f'enc_blk_{n}')

            #deep = tf.reshape(x, (* shape(x)[:-2], shape(x)[-2] * shape(x)[-1]))
            #mask_deep = tf.reshape(mask_x, (* shape(x)[:-2], shape(x)[-2] * shape(x)[-1])) if exists(mask) else None

            ### TODO for single gpu
            #deep, mask_deep = x, mask_x
            ### TODO for multi gpu, more stable
            x = normalization(x, group_size=32, mask=mask_x, trainable=train, scope='group_norm')
            deep, mask_deep = x, mask_x

            """
            # TODO deconv xstride
            x, mask_x = deconvolution(x, 1, dec_features, kernel, stride, padding='same', mask=mask_x, trainable=train, scope='xpan_conv')

            #x = tf.reshape(tf.tile(x[:, :, None], [1, 1, kernel, 1]), [size[0], shape(x)[1] * kernel, enc_features])
            #if exists(mask): mask_x = tf.reshape(tf.tile(mask_x[:, :, None], [1, 1, kernel, 1]), [size[0], shape(mask_x)[1] * kernel, enc_features])
            #x, mask_x = convolution(x, 1, dec_features, kernel, kernel//stride, padding='SAME', mask=mask_x, trainable=train, scope='reproj_conv')

            x = x[:, :size[-2]-P_in]
            if exists(mask): mask_x = mask_x[:, :size[-2]-P_in]

            # TODO upsampolate xstride, h16_r2
            x, mask_x = upsampolate(x, dec_features, axis=1, up_rate=stride, mask=mask_x)

            x = x[:, :size[-2]-P_in]
            if exists(mask): mask_x = mask_x[:, :size[-2] - P_in]

            # TODO upsampolate xstride, h16_r2
            x, mask_x = upsampolate(x, enc_features, axis=1, up_rate=kernel, mask=mask_x)
            x, mask_x = convolution(x, 1, dec_features, 2 * kernel, 2, padding='SAME', mask=mask_x, trainable=train, scope='reproj_conv')

            x = x[:, :size[-2]-P_in]
            if exists(mask): mask_x = mask_x[:, :size[-2] - P_in]
            """

            #x, mask_x = concat_sequence(x, pitch, mask=mask_x, trainable=train, scope='cat_seq')
            #x, mask_x, _ = attn_blk(x, features, mask=mask_x, trainable=train, scope='attn')

            upsize = stride
            dec_dims = [enc_features // int(np.floor(np.sqrt(2) ** n)) for n in range(decodings)]

            reference, mask_ref = linear(reference, out_features=shape(x)[-1], mask=mask_ref, trainable=train, scope='ref_proj')

            x = tf.concat([reference, x], axis=1)
            if exists(mask_x): mask_x = tf.concat([mask_ref, mask_x], axis=1)

            for n in range(decodings):
                #cprint(shape(x)[-1], color='green')
                if upsize != 1:
                    #x = tf.reshape(tf.tile(x[:, :, None], [1, 1, 4, 1]), [size[0], shape(x)[1] * 4, shape(x)[-1]])
                    #if exists(mask): mask_x = tf.reshape(tf.tile(mask_x[:, :, None], [1, 1, 4, 1]), [size[0], shape(mask_x)[1] * 4, shape(mask_x)[-1]])
                    #x, mask_x = convolution(x, 1, dec_features, 8, 2, padding='SAME', mask=mask_x, trainable=train, scope=f'reproj_conv{n}')

                    # TODO decond, h16_fix
                    #x, mask_x = deconvolution(x, 1, max(dec_dims[n], dec_features), 4, 2, padding='same', mask=mask_x, trainable=train, scope=f'xpan_conv{n}')

                    #TODO upsampolate and conv (x4 /2), h16_alt, h8_r1
                    x, mask_x = upsampolate(x, max(dec_dims[n], dec_features), axis=1, up_rate=4, mask=mask_x)
                    x, mask_x = convolution(x, 1, max(dec_dims[n], dec_features), 5, 2, padding='SAME', mask=mask_x, trainable=train, scope=f'reproj_conv{n}')
                    pitch_vec = None

                    upsize /= 2

                else:
                    #x = x[:, :size[-2]]
                    #if exists(mask): mask_x = mask_x[:, :size[-2]]
                    #pitch_vec, _ = convolution(pitch, 1, features//2, 3, 1, padding='SAME', mask=reconstruct_mask(shape(pitch)[-1], mask_x), trainable=train, scope=f'pitch_conv_{n}')
                    pitch_vec = pitch

                """
                if train:
                    reference = tf.reshape(reference, [8, 8, *shape(reference)[-2:]])
                    reference = tf.concat([reference[:, 1:], reference[:, :1]], axis=1)
                    reference = tf.reshape(reference, [64, *shape(reference)[-2:]])
                """

                # TODO 2씩 업샘할 시에
                x, mask_x = decoder_block(x, max(dec_dims[n], dec_features), reference=reference, sequence=pitch if upsize == 1 else None, mask=mask_x, mask_ref=mask_ref, trainable=train, scope=f'dec_blk_{n}')
                # TODO 한 번에 업샘
                #x, mask_x = decoder_block(x, dec_features, latent, mask=mask_x, trainable=train, scope=f'dec_blk_{n}')

            x = x[:, shape(reference)[1]:]
            if exists(mask_x): mask_x = mask_x[:, shape(mask_ref)[1]:]

            x = x[:, :size[-2]]
            if exists(mask): mask_x = mask_x[:, :size[-2]]

        with tf.variable_scope('final'):
            x, mask_x, _ = attn_blk(x, features, mask=mask_x, trainable=train, scope='attn')
            x, mask_x = res_blk(x, features, activation=mish, mask=mask_x, trainable=train, scope='res_0')
            x, mask_x = res_blk(x, features, activation=mish, mask=mask_x, trainable=train, scope='res_1')

            tensor = convolution(x, 1, size[-1], 1, padding='same', mask=mask_x, trainable=train, scope='proj_conv')[0]

        # if translate_weights: TODO implement

        tensor = tf.reshape(tensor, size)
        if exists(mask): tensor *= mask
        #tensor = tf.where(tensor > min_values, tensor, min_values * tf.ones_like(tensor))
        #tensor = tf.nn.relu(tensor)
        #if exists(mask): tensor *= mask
        #tensor = tf.where(tf.equal(mask, 0), min_values * tf.ones_like(tensor), tensor)
        tensor += min_values

        #with tf.control_dependencies([tf_print('generated', tensor, color='magenta')]): tensor = tf.identity(tensor)

        return tensor, (deep, mask_deep)


