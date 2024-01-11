# import base modules
import os
import re
import sys
import glob
import json
import fire
import time
import random
import torch
import itertools
import importlib

# import main modules
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import soundfile as sf

from einops import repeat, rearrange

# import functions only
from pydub import AudioSegment
from tqdm import tqdm
from operator import add
from pprint import pprint
from itertools import cycle
from termcolor import colored, cprint

# import outer custom modules
from .hyper_params import hyper_params as hp
from ..sources import gen_net as G
from ..sources.utils import tf_print, colorize

from ..nsf_hifigan.hparams import hparams, set_hparams


GEN = hp.generator



def value_to_map_tf(value,
                    depth: tf.Tensor | int,
                    value_min: int | float | None = None,
                    value_max: int | float | None = None):
    value_min = tf.reduce_min(value) if value_min is None else value_min
    value_max = tf.reduce_max(value) if value_max is None else value_max
    value = tf.to_float(depth-1)*tf.clip_by_value(
        (value - value_min)/(value_max - value_min), 0., 1.)
    whole = tf.to_int32(value)
    decimal = (value - tf.to_float(whole))[..., None]
    output = tf.one_hot(whole, depth, 1.) * (1.-decimal) + tf.one_hot(whole+1, depth, 1.) * decimal
    return output



class Solver_encoder(Solver):
    def __init__(
            self,
            batch_size,
            label_size,
            # mel-spectrogram
            meltime_size_source,
            meltime_size_target,
            melfreq_size,
            # signal
            sampling_rate,
            window_size,
            hop_size,
            fft_size,
            low_clip_hertz,
            high_clip_hertz,
            #scale_log,
            # optimizer
            learning_rate,
            adam_beta_1=0.0,
            adam_beta_2=0.99,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            decay_steps=None,
            decay_alpha=None,
            ):
        super().__init__(
                batch_size,
                # mel-spectrogram
                meltime_size_source,
                meltime_size_target,
                melfreq_size,
                # signal
                sampling_rate,
                window_size,
                hop_size,
                fft_size,
                low_clip_hertz,
                high_clip_hertz,
                #scale_log,
                # optimizer
                learning_rate,
                adam_beta_1,
                adam_beta_2,
                adam_epsilon,
                weight_decay,
                decay_steps,
                decay_alpha,
                )
        self.N = label_size

        self.n_kwargs = NEUT.copy()
        del self.n_kwargs['load_case']
        self.n_kwargs['train'] = False
        self.n_kwargs['reuse'] = tf.AUTO_REUSE

        self.d_kwargs = DISC.copy()
        del self.d_kwargs['load_case']
        self.d_kwargs['train'] = False
        self.d_kwargs['reuse'] = tf.AUTO_REUSE

        self.g_kwargs = GEN.copy()
        del self.g_kwargs['load_case']
        self.g_kwargs['train'] = False
        self.g_kwargs['reuse'] = tf.AUTO_REUSE

        #self.e_kwargs = ENC.copy()
        #del self.e_kwargs['load_case']
        #self.e_kwargs['train'] = False
        #self.e_kwargs['reuse'] = tf.AUTO_REUSE

        #self.m_kwargs = MAP.copy()
        #del self.m_kwargs['load_case']
        #self.m_kwargs['train'] = False
        #self.m_kwargs['reuse'] = tf.AUTO_REUSE

        self.load_modules = [
                #ENC,
                #'repr_to_vec',
                ]

        self.save_modules = [
                NEUT.scope,
                DISC.scope,
                GEN.scope,
                #ENC.scope,
                #MAP.scope,
                ]


    def __call__(
            self,
            #audio,
            mel,
            latents,
            #nums,
            mlen=None,
            #pitch_coeff=1.0,
            ):
        #_, mel = self.sig_to_mag_mel(audio)     # b * e, l, c

        if exists(mlen):
            #mlen = self.convert_size(signal_size=wlen)
            lmask = tf.sequence_mask(mlen, shape(mel)[1], dtype=tf.float32)

        self.g_kwargs['min_values'] = self.MinMel

        # get pitch components b * e, l, p
        #_, pitch_cmpnts = self.cqt_deconv_tf(audio, fmin=40 * pitch_coeff, fmax=640 * pitch_coeff, bins_per_octave=12)
        pitch_cmpnts = None

        mel_gen = []

        #if not exists(len_cnt): len_cnt = shape(mel_cnt)[1]
        #nums = mlen_cnt // self.T_s + int(mlen_cnt % self.T_s > 0)
        #for n in range(nums):
        #slice_mask = lmask[:, n * self.T_s: tf.minimum((n+1) * self.T_s, shape(mel)[1]), None] if exists(wlen) else None
        mel_gen, _ = G.network(
                #mel[:, n * self.T_s: tf.minimum((n+1) * self.T_s, shape(mel)[1])],
                mel,
                latents,
                pitch=pitch_cmpnts,
                #mask=slice_mask,
                mask=lmask[Ellipsis, None] if exists(mlen) else None,
                **self.g_kwargs)
        #mel_gen.append(slice_gen)

        #mel_gen = tf.concat(mel_gen, axis=1)
        #if exists(mask_cnt): mel_gen = tf.where(tf.equal(mask_cnt, 0.0), self.MinMel * tf.ones_like(mel_gen), mel_gen)

        return mel_gen, mel


    def inputs(self):
        input_holders = [
                tf.placeholder(tf.float32, (None, None, None), 'signal'),
                tf.placeholder(tf.float32, (None, None), 'signal_length'),
                tf.placeholder(tf.float32, (None, None, None), 'signal_f0'),
                tf.placeholder(tf.int32, (None, None), 'mel_length'),
                tf.placeholder(tf.string, (None), 'signal_style_number'),
                tf.placeholder(tf.int32, (None, self.N), 'signal_style_indicator'),
                ]
        return input_holders


    def cqt_deconv_tf(
            self,
            wave,
            fmin,
            fmax,
            bins_per_octave = 12,
            window = "hann"
            ):
        with tf.name_scope('CQTDeconv'):
            from nnAudio.utils import create_cqt_kernels

            # creating kernels for CQT
            Q = 1. / (2 ** (1 / bins_per_octave) - 1)

            cqt_kernels, widths, lengths, _ = create_cqt_kernels(Q, self.SR, fmin, None, bins_per_octave, 1, window, fmax)
            P = (widths - self.H) // 2
            wave = tf.pad(wave, ((0, 0), (P, -shape(wave)[1] % self.H + P)))

            # CQT
            cqt_real = tf.nn.conv1d(wave[Ellipsis, None], cqt_kernels.real.T[:, None], stride=self.H, padding='VALID')
            cqt_imag = -tf.nn.conv1d(wave[Ellipsis, None], cqt_kernels.imag.T[:, None], stride=self.H, padding='VALID')

        # Getting CQT Amplitude
        cqt_spec = tf.sqrt(lengths.numpy()[None, None] * (cqt_real ** 2 + cqt_imag ** 2))

        cqt_max_bin = np.round(bins_per_octave * np.log2(fmax / fmin)).astype(int)

        n_freq = shape(cqt_spec)[-1]
        # Compute the Fourier transform of every frame and their magnitude
        ftcqt_spectrogram = tf.signal.rfft(cqt_spec, [2 * n_freq - 1])
        absftcqt_spectrogram = tf.to_complex64(tf.abs(ftcqt_spectrogram))

        # Derive the spectral component and the pitch component
        spectral_component = tf.signal.irfft(absftcqt_spectrogram)[Ellipsis, :cqt_max_bin]
        pitch_component = tf.signal.irfft(ftcqt_spectrogram / (absftcqt_spectrogram + 1e-16))[Ellipsis, :cqt_max_bin]

        return spectral_component, pitch_component


    def cosine_contrast_loss(
            self,
            tensor,
            reverse=False,
            weight=1.0,
            temperature=0.1,
            ):
        b, e, _ = shape(tensor)

        sim = tf.eye(b, dtype=tf.float32)[Ellipsis, None, None]
        diff = sim - 1.0
        sim *= tf.ones((e, e), dtype=tf.float32)[None, None] # batchs, batchs, enrolls, enrolls
        diff *= tf.ones((e, e), dtype=tf.float32)[None, None]
        answer = tf.reshape(tf.transpose(sim + diff, [0, 2, 1, 3]), [b*e, b*e])
        if reverse: answer = - answer

        tensor = tf.reshape(tensor, [b*e, -1])
        normed = p_normalize(tensor, p=2, axis=-1) #, epsilon=1e-4)     # batchs * enrolls, features

        sim_table = tf.einsum("nf,mf->nm", normed, normed)
        with tf.control_dependencies([
            tf_print('sim_table', sim_table, color='cyan')
            ]):
            sim_table = tf.identity(sim_table)

        ## L2 loss
        #loss = weight * tf.reduce_mean(sim_table - answer)

        ## entropy loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(answer), logits=sim_table)
        loss = weight * tf.reduce_mean(loss)

        ## info NCE loss # TODO not be trained well TODO
        #neg_mask = 1 - repeat(tf.linalg.diag(tf.eye(b, dtype=tensor.dtype), 'm n -> (m a) (m b)', a=e, b=e)
        #splits = tf.split(sim_table, self.B, axis=0)
        #splits =
        #loss = []
        #for n, tab in enumerate(splits):
        #    # tab -> e, b*e
        #    nf, p, nb = tab[:, :n*e], tab[:, n*e:(n+1)*e], tab[:, (n+1)*e:]
        #    n = tf.concat([nf, nb], axis=1) # e, (b-1)*e
        #    p *= 1 - tf.eye(e, dtype=tensor.dtype) # masking equiv
        #    p = tf.reduce_sum(p, axis=1, keepdims=True) / (e - 1) # e * e -> e 1
        #    l = tf.reduce_logsumexp(tf.concat([p, n], axis=1) / temperature, axis=1)
        #    l -= tf.reduce_logsumexp(p / temperature, axis=1)
        #    loss.append(tf.reduce_mean(l))
        #
        #loss = weight * tf.reduce_mean(loss)

        #vals = tf.nn.softmax(sim_table / temperature, axis=-1)
        #mask = repeat(tf.linalg.diag(tf.eye(b, dtype=tensor.dtype)), '... m n -> ... (m a) (n b)', a=e, b=e) # b, b*e, b*e
        #vals = tf.reduce_sum(vals[None] * mask, axis=-1) # b, b*e
        #mask = tf.reduce_sum(mask, axis=-1) # b, b*e
        ##with tf.control_dependencies([
        ##    tf_print('conts', tf.where(tf.equal(mask, 0.), tf.reduce_max(vals) * tf.ones_like(vals), vals), color='cyan'),
        ##    ]):
        ##    vals = tf.identity(vals)
        #loss = tf.reduce_mean(tf.where(tf.equal(mask, 0.), tf.zeros_like(vals), - tf.log(vals)))

        return loss, sim_table


    def cosine_neutralize_loss(
            self,
            tensor,
            weight=1.0,
            temperature=0.1
            ):
        b, e, _ = shape(tensor)

        answer = tf.ones([b*e, b*e], dtype=tf.float32)

        tensor = tf.reshape(tensor, [b*e, -1])
        normed = p_normalize(tensor, p=2, axis=-1) #, epsilon=1e-4)

        sim_table = tf.einsum("nf,mf->nm", normed, normed)
        #with tf.control_dependencies([
        #    tf_print('sim_table', sim_table, color='blue')
        #    ]):
        #    sim_table = tf.identity(sim_table)

        ## L2 loss
        #loss = weight * tf.reduce_mean(tf.square(sim_table - answer))

        ## entropy loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(answer), logits=sim_table)
        loss = weight * tf.reduce_mean(loss)

        ## info NCE loss
        #vals = tf.nn.softmax(sim_table / temperature, axis=-1) # b*e, b*e
        #mask = repeat(tf.linalg.diag(tf.eye(b, dtype=tensor.dtype)), '... m n -> ... (m a) (n b)', a=e, b=e) # b, b*e, b*e
        #vals = tf.reduce_sum(vals[None] * (1 - mask), axis=-1) # b, b*e
        #mask = tf.reduce_sum(mask, axis=-1)
        #with tf.control_dependencies([
        #    tf_print('neuts', tf.where(tf.equal(mask, 0.), tf.reduce_min(vals) * tf.ones_like(vals), vals), color='magenta'),
        #    ]):
        #    vals = tf.identity(vals)
        #loss = tf.reduce_mean(tf.where(tf.equal(mask, 0.), tf.zeros_like(vals), -tf.log(vals)))

        return loss, sim_table


    def style_convert_loss(
            self,
            style_a,
            style_b,
            weight=1.0,
            ):
        normed_a = p_normalize(style_a, p=2, axis=-1)
        normed_b = p_normalize(style_b, p=2, axis=-1)

        table = tf.einsum('nf,nf->n', normed_a, normed_b)
        answer = tf.ones_like(table, dtype=tf.float32)

        #loss = weight * tf.reduce_mean(tf.square(table - answer))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(answer), logits=table)
        loss = weight * tf.reduce_mean(loss)

        return loss, table

    def binary_cross_entropy_loss(
            self,
            tensor,
            labels=None,
            real=True,
            weight=1.0,
            ):
        if labels is None:
            labels = tf.ones_like(tensor, dtype=tensor.dtype) if real else tf.zeros_like(tensor, dtype=tensor.dtype)
        else:
            labels = tf.cast(labels, tensor.dtype)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=tensor)
        loss = weight * tf.reduce_mean(loss)

        return loss

    def disc_grad_penalty(
            self,
            image,
            score,
            mask=None,
            weight=1.0,
            ):
        grads = tf.gradients(tf.reduce_sum(score), [image])[0]
        if exists(mask):
            div = tf.einsum('')


    def feature_matching(
            self,
            t_a,
            t_b,
            masks=None,
            weight=1.0,
            name='feat_match'
            ):
        dict_ = {}
        for n in range(len(t_a)):
            delta = t_a[n] - t_b[n]
            #with tf.control_dependencies([
            #    tf_print(f't_a_{n}', t_a[n], color='cyan'),
            #    tf_print(f't_b_{n}', t_b[n], color='yellow'),
            #    tf_print(f'delta_{n}', delta, color='green')
            #    ]): delta = tf.identity(delta)
            sq = (delta ** 2)

            if exists(masks):
                mse = tf.reduce_sum(sq, axis=range(1, len(shape(t_a[n]))))
                mse /= tf.maximum(tf.reduce_sum(masks[n], axis=range(1, len(shape(t_a[n])))), 1e-8)
            else:
                mse = tf.reduce_mean(sq)

            dict_[f'{name}_{n}'] = weight * tf.reduce_mean(mse)

        return dict_


    def build_graph(
            self,
            signals,
            signal_lengths,
            signal_f0s,
            mel_lengths,
            speaker_numbers,
            speaker_onehots,
            ):

        train_status = get_current_tower_context()
        start_time = time.time()
        local = locals()

        with tf.variable_scope('init_graph'):
            b, e, _ = shape(signals)
            signals = tf.reshape(signals, [b * e, -1])
            #signal_lengths = tf.reshape(signal_lengths, [b * e])
            mel_lengths = tf.reshape(mel_lengths, [b * e])

            _, mels = self.sig_to_mag_mel(signals)     # b * e, l, c
            mels = tf.reshape(mels, [b, e] + shape(mels)[1:])

            ### mask via signal lenths
            #masks = tf.sequence_mask(self.convert_size(signal_size=signal_lengths), shape(mels)[-2], dtype=tf.float32)

            ### mask via array lengths
            masks = tf.sequence_mask(mel_lengths, shape(mels)[-2], dtype=tf.float32)

            masks = tf.tile(masks[Ellipsis, None], [1, 1, self.M])
            masks = tf.reshape(masks, [b, e] + shape(masks)[1:])


        self.d_kwargs['train'] = train_status
        self.n_kwargs['train'] = train_status

        self.g_kwargs['min_values'] = self.MinMel
        self.g_kwargs['train'] = train_status

        if getattr(self, 'generator', None) is None:
            self.generator = G.Network(**self.g_kwargs)
        else:
            self.generator.reset_states()

        #self.e_kwargs['min_values'] = self.MinMel
        #self.e_kwargs['train'] = train_status

        #self.m_kwargs['train'] = train_status

        mels_real = tf.reshape(mels, [-1] + shape(mels)[-2:])
        masks_real = tf.reshape(masks, [-1] + shape(masks)[-2:])

        mels_transpose = tf.reshape(tf.transpose(mels, [1, 0, 2, 3]), [b * e, *shape(mels)[-2:]])
        masks_transpose = tf.reshape(tf.transpose(masks, [1, 0, 2, 3]), [b * e, *shape(masks)[-2:]])

        # get style reprs
        #style_real = E.network(mels_real, mask=masks_real, **self.e_kwargs)
        # batchs * enrolls, ref_length, features

        #style_inputs = tf.transpose(tf.reshape(style_real, [b, e, -1, ENC.features]), [1, 0, 2, 3])
        #style_inputs = tf.reshape(style_inputs, shape(style_real))

        #latents = M.network(style_inputs, **self.m_kwargs)

        # get pitch components b * e, l, p
        #_, pitch_cmpnts_base = self.cqt_deconv_tf(signals, fmin=40, fmax=640, bins_per_octave=12)
        #_, pitch_cmpnts = self.cqt_deconv_tf(signals, fmin=20, fmax=1280, bins_per_octave=12)[Ellipsis, 12:-12] # orig pitch
        #_, pitch_cmpnts_lower = self.cqt_deconv_tf(signals, fmin=80, fmax=1280, bins_per_octave=12) # high to low pitch
        #_, pitch_cmpnts_higher = self.cqt_deconv_tf(signals, fmin=20, fmax=320, bins_per_octave=12) # low to high pitch

        """
        #signal_f0s = tf.reshape(signal_f0s, [b * e, -1])[Ellipsis, None]     # [b, e, l] -> [b*e, l, 1]
        signal_f0s = signal_f0s[Ellipsis, None]
        pitch_cmpnts_base = signal_f0s
        pitch_cmpnts_lower = signal_f0s #* 0.5
        pitch_cmpnts_higher = signal_f0s #* 2.0

        b_ran, e_ran = tf.range(b)[:, None], tf.range(e)[None]
        #diag_mask = tf.reshape(tf.cast(b_ran == e_ran, tf.float32), [-1])[:, None, None]
        #low_mask = tf.reshape(tf.cast(b_ran < e_ran, tf.float32), [-1])[:, None, None]
        #high_mask = tf.reshape(tf.cast(b_ran > e_ran, tf.float32), [-1])[:, None, None]
        diag_mask = tf.cast(b_ran == e_ran, tf.float32)[Ellipsis, None, None]
        low_mask = tf.cast(b_ran < e_ran, tf.float32)[Ellipsis, None, None]
        high_mask = tf.cast(b_ran > e_ran, tf.float32)[Ellipsis, None, None]

        pitch_cmpnts = diag_mask * pitch_cmpnts_base + low_mask * pitch_cmpnts_lower + high_mask * pitch_cmpnts_higher
        pitch_cmpnts = tf.reshape(pitch_cmpnts, [b * e, -1, 1])
        #pitch_cmpnts *= tf.to_float(pitch_cmpnts > 0.3)
        """
        pitch_cmpnts = tf.reshape(signal_f0s, [b * e, -1])[Ellipsis, None]

        ### update restrictors
        reviewers_ = dict()

        # transferred image
        #mels_trsf, (deep, mask_deep) = G.network(mels_real, latents, pitch_cmpnts, mask=masks_real, **self.g_kwargs)
        mels_trsf, (deep, mask_deep) = self.generator(
                mels_real, mels_transpose, pitch_cmpnts,
                mask_source = masks_real, mask_reference = masks_transpose,
                train=self.g_kwargs['train'], reuse=self.g_kwargs['reuse'])

        sorted_deep, sorted_deep_mask = map(lambda t: tf.reshape(t, [b, e, -1, shape(t)[-1]]), (deep, mask_deep))

        start_index = tf.random.uniform([], minval=1, maxval=e, dtype=tf.int32)

        def get_pair_by_index(t, id): return tf.stack([t, tf.concat([t[:, id:], t[:, :id]], axis=1)], axis=-1)
        positive_deep_pairs = get_pair_by_index(sorted_deep, start_index)
        positive_deep_mask_pairs = get_pair_by_index(sorted_deep_mask, start_index)
        # b, e, l, c, 2 (each last dim denotes 2 different instances from a speaker)

        transp_deep, transp_deep_mask = map(lambda t: tf.transpose(t, [1, 0, 2, 3]), (sorted_deep, sorted_deep_mask))
        negative_deep_pairs = get_pair_by_index(transp_deep, start_index)
        negative_deep_mask_pairs = get_pair_by_index(transp_deep_mask, start_index)

        pos_dps, pos_dms = map(lambda t: tf.reshape(t, [b*e, *shape(t)[2:]]), (positive_deep_pairs, positive_deep_mask_pairs))
        neg_dps, neg_dms = map(lambda t: tf.reshape(t, [b*e, *shape(t)[2:]]), (negative_deep_pairs, negative_deep_mask_pairs))

        v_positives, _ = N.multi_network(pos_dps, mask=pos_dms, **self.n_kwargs)
        v_negatives, _ = N.multi_network(neg_dps, mask=neg_dms, **self.n_kwargs)

        loss_neut_pos = self.binary_cross_entropy_loss(v_positives, real=True, weight=1.0)
        loss_neut_neg = self.binary_cross_entropy_loss(v_negatives, real=False, weight=1.0)

        reviewers_['deep_pos'] = loss_neut_pos
        reviewers_['deep_neg'] = loss_neut_neg

        # training quality verifier
        v_real, _ = D.multi_network(mels_real, mask=masks_real, **self.d_kwargs)
        v_fake, _ = D.multi_network(mels_trsf, mask=masks_real, **self.d_kwargs)

        loss_disc_r = self.binary_cross_entropy_loss(v_real, real=True, weight=1.0)
        loss_disc_f = self.binary_cross_entropy_loss(v_fake, real=False, weight=1.0)

        reviewers_['disc_real'] = loss_disc_r
        reviewers_['disc_fake'] = loss_disc_f

        # total loss
        reviewers_['total'] = tf.reduce_sum(list(reviewers_.values()))

        #cprint(reviewers_, color='red')


        ### update generating modules
        generate_ = dict()

        # deep style neutralization
        _, (feats_deep_pos, fmasks_deep_pos) = N.multi_network(pos_dps, mask=pos_dms, **self.n_kwargs)
        v_orthogonals, (feats_deep_neg, fmasks_deep_neg) = N.multi_network(neg_dps, mask=neg_dms, **self.n_kwargs)

        loss_neut_adv = self.binary_cross_entropy_loss(v_orthogonals, real=True, weight=5.0)
        # cast_0 0.1 case_1 0.05 cast_2 0.01

        generate_['neut_adv'] = loss_neut_adv

        Deep_Feat_Match = self.feature_matching(
                #tf.stop_gradient(
                feats_deep_pos,#),
                feats_deep_neg,
                name='neut_feat_match',
                weight=0.27
                )

        generate_.update(Deep_Feat_Match)

        #mels_trsf = tf.reshape(mels_trsf, [b, e, -1, self.M])
        # content_restriction
        reals = tf.einsum("belm,be->blm", mels, tf.eye(b, dtype=tf.float32))
        #fakes = tf.einsum("belm,be->blm", mels_trsf, tf.eye(b, dtype=tf.float32))
        masks_cnt = tf.einsum("belm,be->blm", masks, tf.eye(b, dtype=tf.float32))

        _, (feats_r, fmasks_r) = D.multi_network(reals, mask=masks_cnt, **self.d_kwargs)
        #_, (feats_f, fmasks_f) = D.multi_network(fakes, mask=masks_cnt, **self.d_kwargs)
        v_gen, (whole_feats_f, whole_fmasks_f) = D.multi_network(mels_trsf, mask=masks_real, **self.d_kwargs)
        #v_gen = tf.reshape(v_gen, [b, e, *shape(v_gen)[1:]])
        #whole_fmasks_f whole_fmasks_f
        #v_gen = tf.reshape(v_gen * (1.0 - tf.eye(b, dtype=tf.float32))[Ellipsis, None], [b * e, *shape(v_gen)[2:]])

        loss_disc_g = self.binary_cross_entropy_loss(v_gen, real=True, weight=1.0) #tf.where(self.global_step > 100000, 1.0, 0.0))
        generate_['disc_gen'] = loss_disc_g


        feats_f = list(map(lambda t: tf.einsum("belmd, be-> blmd", tf.reshape(t, [b, e, *shape(t)[1:]]), tf.eye(b, dtype=tf.float32)), whole_feats_f))
        #feats_f, fmasks_f = map(lambda t: tf.einsum("belm, be->blm", t, tf.eye(b, dtype=tf.float32)), (whole_feats_f, whole_fmasks_f))

        Feat_Match = self.feature_matching(
                feats_f,
                feats_r,
                fmasks_r,
                name='cnt_feat_match',
                weight=50.0
                )

        generate_.update(Feat_Match)

        generate_['total'] = tf.reduce_sum(list(generate_.values()))

        #cprint(generate_, color='red')

        with tf.name_scope('loss'):
            for key in reviewers_.keys(): local['loss_RESTR_' + key] = reviewers_[key]
            for key in generate_.keys(): local['loss_GEN_' + key] = generate_[key]

        [tf.add_to_collection(module, var) for module in self.save_modules for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, f'^.*{module}/')]

        with tf.name_scope('train_nets'):
            #disp_pit_base, disp_pit_low, disp_pit_high = map(lambda p: tf.reshape(p, [b, e, *shape(pitch_cmpnts)[1:]]), (pitch_cmpnts_base, pitch_cmpnts_lower, pitch_cmpnts_higher))
            #disp_pit_base, disp_pit_low, disp_pit_high = map(lambda p: tf.sequence_mask(tf.squeeze(p, axis=-1), maxlen = tf.cast(hp.pitch_f0_max, tf.int32), dtype=tf.float32),
            #        (pitch_cmpnts_base, pitch_cmpnts_lower, pitch_cmpnts_higher))
                    #(disp_pit_base, disp_pit_low, disp_pit_high))
            #disp_pit_base, disp_pit_low, disp_pit_high = map(lambda p: p[Ellipsis, 1:] - p[Ellipsis, :-1], (disp_pit_base, disp_pit_low, disp_pit_high))

            #disp_pit_base, disp_pit_low, disp_pit_high = map(lambda p: value_to_map_tf(p, depth=32, value_min=40.0, value_max=1100.0), (pitch_cmpnts_base, pitch_cmpnts_lower, pitch_cmpnts_higher))


            # TODO 여기 summary, return loss
            tf.summary.image('00_real', colorize(mels[:1, 0], self.MinMel, self.MinMel + 6), 1)
            tf.summary.image('01_real', colorize(mels[:1, 1], self.MinMel, self.MinMel + 6), 1)
            tf.summary.image('10_real', colorize(mels[1, :1], self.MinMel, self.MinMel + 6), 1)

            mels_trsf = tf.reshape(mels_trsf, [b, e, *shape(mels_trsf)[1:]])
            tf.summary.image('00_00_pitch_base', colorize(mels_trsf[:1, 0], self.MinMel, self.MinMel + 6), 1)
            tf.summary.image('01_10_pitch_low', colorize(mels_trsf[:1, 1], self.MinMel, self.MinMel + 6), 1)
            tf.summary.image('10_01_pitch_high', colorize(mels_trsf[1, :1], self.MinMel, self.MinMel + 6), 1)

            pitch_cmpnts_base = signal_f0s[Ellipsis, None]
            tf.summary.image('pitch_base_0-0', colorize(value_to_map_tf(pitch_cmpnts_base[:1, 0, :, 0], depth=32, value_min=40.0, value_max=1100.0), 0.0, 1.0), 1)
            tf.summary.image('pitch_base_0-1', colorize(value_to_map_tf(pitch_cmpnts_base[:1, 1, :, 0], depth=32, value_min=40.0, value_max=1100.0), 0.0, 1.0), 1)
            tf.summary.image('pitch_base_1-0', colorize(value_to_map_tf(pitch_cmpnts_base[1, :1, :, 0], depth=32, value_min=40.0, value_max=1100.0), 0.0, 1.0), 1)
            for name in local:
                if name.startswith('loss'): tf.summary.scalar(name, local[name])

        return (
                [
                    local['loss_RESTR_total'],
                    local['loss_GEN_total'],
                    ],
                [
                    [
                        NEUT.scope,
                        #DEXT.scope,
                        #DREC.scope,
                        DISC.scope,
                        ],
                    [
                        #ENC.scope,
                        GEN.scope,
                        #MAP.scope
                        ],
                    ]
                )


def train(gpu, case, transfer=False):

    num_gpu = len(gpu)
    if num_gpu > 1:
        cprint('parallel-machine training', color='red')
        trainer = Multi_encoder(gpus=num_gpu, penalty_targets=hp.penalty_targets, penalty_rate=hp.penalty_rate)
    else:
        cprint('single-machine training', color='red')
        trainer = Single_encoder(penalty_targets=hp.penalty_targets, penalty_rate=hp.penalty_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    spks = list(map(lambda f: f.split('/')[-1], glob.glob(os.path.join(hp.train_path, '*'))))
    speakers, depth = [], 0
    while speakers == []:
        speakers = [spk for spk in spks if len(glob.glob(os.path.join(hp.train_path, spk, ('*/'*depth) + '*.wav')))
                >= hp.batch_size]
        depth += 1

    cprint(f'among {len(spks)} total speakers, {len(speakers)} speakers has chosen'
    + f' since sample amounts are greater or equall to {hp.batch_size}', 'yellow')

    dataset = Dataset_encoder(
            data_path=hp.train_path, depth=depth, speakers=speakers,
            batch_size=hp.batch_size,
            enrollment_size=hp.batch_size, sampling_rate=hp.sampling_rate, window_size=hp.window_size,
            hop_size=hp.hop_size, fft_size=hp.fft_size, meltime_size=hp.meltime_size, melfreq_size=hp.melfreq_size,
            f0_min=hp.pitch_f0_min, f0_max=hp.pitch_f0_max)

    train_dataflow, n_total_data, n_train_data = dataset(
            mode='train', data_list=dataset.speakers, split_ratio=0., n_prefetch=hp.n_prefetch, n_process=hp.n_process,
            scope='train_dataset')

    solver = Solver_encoder(
            batch_size=hp.batch_size,
            label_size=len(speakers),
            meltime_size_source=hp.meltime_size, meltime_size_target=hp.meltime_size,
            melfreq_size=hp.melfreq_size, sampling_rate=hp.sampling_rate,
            window_size=hp.window_size, hop_size=hp.hop_size, fft_size=hp.fft_size,
            low_clip_hertz=hp.low_clip_hertz, high_clip_hertz=hp.high_clip_hertz,
            learning_rate=hp.learning_rate,
            adam_beta_1=hp.adam_beta_1, adam_beta_2=hp.adam_beta_2, adam_epsilon=hp.adam_epsilon,
            weight_decay=hp.weight_decay, decay_steps=hp.decay_steps, decay_alpha=hp.decay_alpha)

    log_location = os.path.join(hp.pack_path, hp.log_path, case)
    logger.set_logger_dir(log_location)

    save_dirs = list(map(lambda module: os.path.join(log_location, os.path.basename(module)), solver.save_modules))

    callbacks = list(map(lambda save_dir, module: ModelSaver(5, 2, save_dir, [tf.GraphKeys.GLOBAL_STEP, module]),
        save_dirs, solver.save_modules))

    current_checkpoints = list(filter(lambda c: c is not None, map(tf.train.latest_checkpoint, save_dirs)))
    cprint(current_checkpoints, color='yellow')

    init_args = [current_checkpoints]

    if transfer:
        init_args += [[None] * len(current_checkpoints), [['global_step']] * len(current_checkpoints)]

    inits = list(map(SaverRestore, *init_args)) if current_checkpoints else []

    if isinstance(solver.load_modules, list) or isinstance(solver.load_modules, tuple):
        if len(solver.load_modules) != 0:
            cprint(solver.load_modules, color='magenta')

            load_dirs = list(map(lambda module: os.path.join(hp.pack_path, hp.log_path, module.load_case, module.scope),
                solver.load_modules))
            cprint(load_dirs, color='magenta')

            pre_trained_checkpoints = list(filter(lambda c: exists(c), map(tf.train.latest_checkpoint, load_dirs)))
            cprint(pre_trained_checkpoints, color='magenta')

            load_args = [pre_trained_checkpoints]
            load_args += [[None] * len(pre_trained_checkpoints), [['global_step']] * len(pre_trained_checkpoints)]

            loads = list(map(SaverRestore, *load_args)) if pre_trained_checkpoints else []

            inits += loads

    train_config = TrainConfig(model=solver, dataflow=train_dataflow, callbacks=callbacks,
            steps_per_epoch=hp.steps_per_epoch, session_init=(ChainInit(inits) if inits else None),
            monitors=[TFEventWriter(), JSONWriter(), ScalarPrinter()],)

    launch_train_with_config(train_config, trainer=trainer)



