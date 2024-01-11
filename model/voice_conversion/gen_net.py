import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .Neural_Networks import *




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
            res_out, res_mask = convolution(res_out, 1, 4 * features, 7, mask=res_mask, quantization=R_Q, quantization_blocks=7, trainable=trainable, scope='conv1d_0')
            res_out *= (7 ** -0.5)
            res_out = activation(res_out) * res_mask if exists(mask) else activation(res_out)
            if trainable: res_out = tf.nn.dropout(res_out, rate=R_D, name='drop')
            res_out, res_mask = convolution(res_out, 1, features, 3, mask=res_mask, trainable=trainable, scope='conv1d_1')
            res_out *= (3 ** -0.5)

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        tensor = (lambda a, b: a+ b)(*map(lambda a, b: a * b, (res_out, sc_out), res_scale(0.9, tensor.dtype, trainable=trainable, scope='coeff')))
        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask



def attn_blk(
        tensor,
        features,
        saved_state=None,
        mask=None,
        trainable=True,
        scope='attn_blk'
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')

            res_out, res_mask, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask, saved_state=saved_state,
                    quantization=R_Q, quantization_blocks=features // 4,
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

        tensor = (lambda a, b: a+ b)(*map(lambda a, b: a * b, (res_out, sc_out), res_scale(0.9, tensor.dtype, trainable=trainable, scope='coeff')))
        if exists(mask): tensor *= sc_mask

        return tensor, sc_mask, res_out



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

        return tensor, mask



def decoder_block(
        tensor,
        reference,
        features,
        sequence,
        saved_states=None,
        mask=None,
        mask_reference=None,
        mask_sequence=None,
        trainable=True,
        scope='dec_blk'
        ):
    with tf.variable_scope(scope):
        seq, _ = convolution(sequence, 1, features, 3, mask=mask_sequence, trainable=trainable, scope='seq_conv')
        tensor += seq

        attn_state = getattr(saved_states, 'attn', StateSaver())
        tensor, mask, _ = attn_blk(tensor, features, mask=mask, saved_state=attn_state, trainable=trainable, scope='attn')

        adattn_state = getattr(saved_states, 'ada_attn', StateSaver())
        tensor, mask = adaptive_attention(tensor, reference, out_features=features, heads=features // 64, group_size=16,
                    saved_state=adattn_state, mask_tensor=mask, mask_reference=mask_reference, quantization=R_Q,
                    quantization_blocks=features // 4, scale=False, shift=False, trainable=trainable, scope='ada_attn')

        tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res_ff')

        return tensor, mask



def pitch_encoder(
        tensor,
        features,
        mask = None,
        train = True,
        scope = 'pit_enc',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):

        pitch_feat, pitch_mask = linear(tensor, out_features = 2 * features, mask = mask, trainable = train, scope = 'pitch_proj_0')
        gate, filt = tf.split(pitch_feat, 2, axis=-1)
        pitch_feat = tf.sigmoid(gate) * tf.tanh(filt)
        if exists(mask):
            pitch_mask = pitch_mask[Ellipsis, :features]
            pitch_feat *= pitch_mask
        pitch_feat, _ = linear(pitch_feat, out_features = 2 * features, mask = pitch_mask, quantization = R_Q, quantization_blocks = features, trainable = train, scope = 'pitch_proj_1')
        gate, filt = tf.split(pitch_feat, 2, axis=-1)
        pitch_feat = tf.sigmoid(gate) * tf.tanh(filt)
        if exists(mask): pitch_feat *= pitch_mask

    return pitch_feat, pitch_mask




def time_pos_emb(
        t,
        features
        ):
    half_feat = features // 2
    emb = tf.log(10000.0) / (half_feat - 1)
    emb = tf.exp(tf.range(half_feat, dtype=t.dtype) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb



def mel_encoder(
        tensor,
        features,
        stride = 1,
        encodings = 3,
        reduction_per_encoding = 1,
        mask = None,
        activation = mish,
        train = True,
        scope = 'mel_enc',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse = reuse):

        with tf.variable_scope('initial'):
            tensor, mask = linear(tensor, out_features=features, mask=mask, trainable=train, scope='proj_lin')
            tensor = normalization(tensor, group_size=16, mask=mask, trainable=train, scope='group_norm')
            tensor = activation(tensor) * mask if exists(mask) else activation(tensor)
            tensor, mask = convolution(tensor, 1, features, 2 * stride, stride, padding='same', mask=mask, trainable=train, scope='proj_conv')

        with tf.variable_scope('central'):
            for n in range(encodings):
                tensor, mask = encoder_block(tensor, features, mask=mask, trainable=train, scope=f'enc_blk_{n}')

                # downsample을 매번 동일한 맥락에서 진행하기 위해 동일한 함수를 적용하되, timestep에 대한 성분을 input에 더해준다.
                #timestep = time_pos_emb(n * tf.ones([shape(tensor)[0]], dtype=tensor.dtype), features)
                #timestep, _ = linear(timestep, out_features=4 * features, trainable=train, scope='timestep_linear_0')
                #timestep, _ = linear(mish(timestep), out_features=features, trainable=train, scope='timestep_linear_1')
                #tensor += timestep[:, None]

                if reduction_per_encoding != 1:
                    tensor, mask = convolution(tensor, 1, features, reduction_per_encoding * 2, reduction_per_encoding, padding='SAME',
                            mask = mask, trainable=train, scope=f'conv_reduction_{n}')
                    tensor = activation(tensor) * mask if exists(mask) else activation(tensor)

    return tensor, mask



def mel_decoder(
        source,
        reference,
        pitch,
        features,
        stride = 1,
        decodings = 3,
        mel_minimum = 0.0,
        mel_shape = None,
        saved_states = None,
        mask_source = None,
        mask_reference = None,
        mask_pitch = None,
        activation = mish,
        train = True,
        scope = 'mel_dec',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse = reuse):
        upsize = stride

        with tf.variable_scope('central'):
            for n in range(decodings):
                up_feats = features
                if upsize != 1:
                    up_feats = features * int(np.ceil(np.sqrt(upsize)))
                    source, mask_source = upsampolate(source, up_feats, axis=1, up_rate=4, mask=mask_source)
                    source, mask_source = convolution(source, 1, up_feats, 9, 2, padding='SAME', mask=mask_source, trainable=train, scope=f'reproj_conv{n}')
                    upsize /= 2
                elif exists(mel_shape):
                    source = source[Ellipsis, :mel_shape[-2], :]
                    if exists(mask_source): mask_source = mask_source[Ellipsis, :mel_shape[-2], :]

                state = getattr(saved_states, f'dec_blk_{n}', StateSaver())
                source, mask_source = decoder_block(
                        source, reference, up_feats, pitch, saved_states = state, mask=mask_source, mask_reference=mask_reference, mask_sequence=mask_pitch,
                        trainable=train, scope=f'dec_blk_{n}')

        with tf.variable_scope('terminal'):
            tensor, mask, _ = attn_blk(source, features, mask=mask_source, trainable=train, scope='attn')
            tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=train, scope='res_0')
            tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=train, scope='res_1')
            tensor = activation(tensor) * mask if exists(mask) else activation(tensor)
            tensor, mask = convolution(tensor, 1, mel_shape[-1], 3, padding='same', mask=mask, trainable=train, scope='proj_conv')

            tensor = tf.reshape(tensor, mel_shape)
            if exists(mask): tensor *= mask
            tensor += mel_minimum

        return tensor



class Network():
    def __init__(
            self,
            features = 128,
            stride = 4,
            encodings_source = 3,
            encodings_reference = 1,
            decodings = 5,
            min_values=0.0,
            scope = 'gen_net',
            **kwargs,
            ):
        self.feats = features
        self.strd = stride
        ratio = int(np.floor(np.sqrt(stride)))
        self.deep_feats = int(features * ratio)
        self.enc_src = encodings_source
        self.enc_ref = encodings_reference
        self.dec = decodings
        self.min_mel = min_values
        self.scope = scope
        self.act = mish

        self.reset_states()

    def __call__(
            self,
            tensor,
            reference,
            pitch,
            mask_source = None,
            mask_reference = None,
            train = True,
            reuse=tf.AUTO_REUSE
            ):
        with tf.variable_scope(self.scope, reuse=reuse):
            size = shape(tensor)

            src, mask_src = mel_encoder(
                    tensor, self.deep_feats, stride = self.strd, encodings = self.enc_src, reduction_per_encoding=1, mask = mask_source,
                    activation = self.act, train = train, scope = 'source_encoder', reuse = reuse)
            #src = normalization(src, group_size=32, mask=mask_src, trainable=train, scope='source_group_norm')

            ref, mask_ref = mel_encoder(
                    reference, self.feats, stride = self.strd, encodings = self.enc_ref, mask = mask_reference, reduction_per_encoding=2,
                    activation = self.act, train = train, scope = 'reference_encoder', reuse = reuse)

            pit, mask_pit = pitch_encoder(
                    pitch, self.feats, mask = reconstruct_mask(1, mask_source, axis=-1), train = train, scope = 'pitch_encoder', reuse = reuse)

            gen = mel_decoder(
                    src, ref, pit, self.feats, stride = self.strd, decodings = self.dec, mel_minimum = self.min_mel, mel_shape = size,
                    saved_states = self.decoded_states, mask_source = mask_src, mask_reference = mask_ref, mask_pitch = mask_pit,
                    activation = self.act, train = train, scope = 'synthesis_decoder', reuse = reuse)

            if train: self.reset_states()

        return gen, (src, mask_src)

    def reset_states(self):
        self.decoded_states = StateSaver()

    def set_distance(self, state, value):
        subclasses = [attr_name for attr_name in dir(state) if not attr_name.starts_with('__')]
        assert len(subclasses) > 0
        for sub_name in subclasses:
            sub_cls = getattr(state, sub_name)
            if isinstance(sub_cls, StateSaver):
                self.set_distance(sub_cls, value)
            setattr(sub_cls, 'distance', value)

