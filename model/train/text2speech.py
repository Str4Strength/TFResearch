import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import soundfile as sf
import wave as wv

from glob import glob
from termcolor import cprint
from tqdm import tqdm
from random import shuffle
from multiprocessing import Pool
from scipy.signal import resample
from collections import Counter

from tensorpack.dataflow import DataFromList, MultiProcessMapDataZMQ
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.sessinit import ChainInit, SaverRestore
from tensorpack.callbacks import JSONWriter, ScalarPrinter, TFEventWriter
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.train.interface import TrainConfig, launch_train_with_config

from tensorflow.python.ops.script_ops import _py_funcs

from ..text2speech import *
from ..utils import *
from ..audio_utils import *

from ..hyper_params import hyper_params as hp

import os
osp = os.path
import re
import subprocess

from .meta import *



AUD = hp.audio

class Dataset():

    def __init__(
            self,
            data_path,
            gpus,
            package_size,
            model_size,
            speakers,
            clusters,
            slicer,
            sampling_rate,
            hop_size,
            #f0_max,
            #f0_min,
            voice_left,
            voice_right,
            dBFS,
            shuffle = False,
            ):
        self.data_path = data_path
        self.gpus = gpus
        self.size = package_size
        self.model_size = model_size
        self.size_factor = None

        self.SR = sampling_rate
        self.H = hop_size
        self.dBFS = dBFS

        wave_paths = set(subprocess.run(
            ["find", osp.join(self.data_path, 'wav'), "-path", osp.join(self.data_path, 'wav', '*'), "-name", "*.wav"],
            stdout = subprocess.PIPE
            ).stdout.decode().splitlines())

        csv_paths = set(subprocess.run(
            ["find", osp.join(self.data_path, 'csv'), "-name", "*.csv"],
            stdout = subprocess.PIPE
            ).stdout.decode().splitlines())

        self.clusters = dict([l.strip().split('|') for l in open(osp.join(self.data_path, 'kmeans', f'{clusters}.csv'))])
        count_cluster = Counter(self.clusters.values())
        available_cluster = {k: [] for k in count_cluster if count_cluster[k] > 1}
        self.cluster_to_meta = {}

        raws = []
        lines = [l.strip() for p in csv_paths for l in open(p, 'r', encoding='utf-8')][::slicer]

        for l in tqdm(lines, ascii = True, dynamic_ncols = True):

            if l.startswith('#'): continue

            speaker, file, language, token_size, total_wave_size, start, end, sampling_rate, script = l.split('|')
            if language != 'ko': continue # TODO aligner 초반에 한국어 화자만 할 것
            token_size = int(token_size)

            #if not re.search(speakers, speaker): continue

            wave_file = osp.join(self.data_path, 'wav', speaker, file + '.wav')
            if wave_file not in wave_paths:
                cprint(f'{wave_file} not exists!', color = 'red')
                continue

            wave_size_multiplier = int(sampling_rate) / self.SR
            start_frame = max(0, int(start) - voice_left)
            end_frame = min(- (- int(total_wave_size) / wave_size_multiplier // self.H), int(end) + voice_right)
            wave_size = int((end_frame - start_frame) * hop_size)

            if wave_size == 0: continue
            #if wave_size > 150 * self.H: continue  # 초기에만

            size_factor = self._calculate_size(wave_size, token_size, self.model_size)
            if size_factor > self.size: continue

            raws += [(speaker, file, language, script, start_frame, end_frame, token_size, wave_size, size_factor)]

        self.raws = raws
        if shuffle:
            shuffle(self.raws)
        else:
            self.raws.sort(key = lambda k: k[-1])

        token_sizes, wave_sizes = [], []

        num_raws = len(self.raws)
        token_size, wave_size = 0, 0

        splitter_start = [0]

        for splitter_end in tqdm(range(num_raws + 1), ascii = True, dynamic_ncols = True):
            minibatch = self.raws[splitter_start[-1] : 1 + splitter_end]
            num_minibatch = len(minibatch)

            if num_minibatch == 1:
                token_size, wave_size = minibatch[-1][-3 : -1]

            else:
                token_size, wave_size = max(token_size, minibatch[-1][-3]), max(wave_size, minibatch[-1][-2])

            size_factor = self._calculate_size(wave_size, token_size, self.model_size)

            if size_factor * num_minibatch > self.size or splitter_end == num_raws:
                b = list(zip(*self.raws[splitter_start[-1] : splitter_end]))
                splitter_start.append(splitter_end)
                token_sizes.append(max(b[-3]))
                wave_sizes.append(max(b[-2]))

        splitters = list(zip(splitter_start[:-1], splitter_start[1:]))
        self.splitters = list(zip(splitters, token_sizes, wave_sizes))

        cprint(len(self.splitters), color = 'cyan')


    def _calculate_size(self, wave_size, token_size, model_size):
        mel_size = - (- wave_size // self.H)
        first_unit = (mel_size ** 2)
        second_unit = mel_size * model_size
        third_unit = (model_size ** 2)
        return first_unit + second_unit + third_unit


    def __call__(
            self,
            threads = 32,
            buffers = 2,
            ):
        splitters = self.splitters.copy()
        if self.gpus > 1:
            splitters = splitters[:len(splitters) - (len(splitters) % self.gpus)]

            chunked = [splitters[i : i + self.gpus] for i in range(0, len(splitters), self.gpus)]
            shuffle(chunked)
            splitters = [x for c in chunked for x in c]

        minibatches = [(list(self.raws[start : end]), token_size, wave_size) for (start, end), token_size, wave_size in splitters if start != end]

        dataset = DataFromList(minibatches, shuffle = self.gpus <= 1)
        dataset = MultiProcessMapDataZMQ(
                ds = dataset,
                num_proc = threads,
                map_func = self._get_input_,
                buffer_size = buffers
                )

        return dataset, len(splitters)


    def _get_input_(
            self,
            minibatches
            ):
        outputs = []
        max_token_size, max_wave_size = minibatches[1:]

        for minib in minibatches[0]:
            speaker, file, language, script, start_frame, end_frame, _, wave_size, _ = minib

            base = osp.join(speaker, file)

            text = cleaner(script, language)
            tokens = np.asarray(tokenize(text, language)[0], dtype = np.int32)

            token_size = np.int32(len(tokens))
            if not token_size: continue

            tokens = np.pad(tokens, (0, max(0, max_token_size - token_size)), 'constant', constant_values = (0, 0))[:max_token_size]

            wave_file = osp.join(self.data_path, 'wav', base + '.wav')
            with wv.open(wave_file, 'rb') as f:
                multiplier = f.getframerate() / self.SR
                try:
                    f.setpos(int(start_frame * multiplier * self.H))
                except:
                    cprint((start_frame, end_frame, multiplier, self.H, f.getframerate(), self.SR), color = 'blue')
                wave = np.frombuffer(f.readframes(int(wave_size * multiplier)), np.int16) / (2 ** 15)
                wave = normalize_audio(wave, self.dBFS)

                if multiplier != 1:
                    wave = resample(wave, int(len(wave) / multiplier))

            wave_size = np.int32(len(wave))
            if max_wave_size - wave_size < 0: print('wave_size:', wave_size, 'max_wave_size:', max_wave_size)
            wave = np.pad(wave, (0, max_wave_size - wave_size))

            speaker_id = self.clusters[base]

            outputs.append((
                #base,
                speaker_id,
                tokens,
                token_size,
                wave,
                wave_size,
                ))

        assert len(outputs) >= 1
        return tuple(zip(*outputs))



class Single_Machine_Trainer(Single_GPU_Trainer, MakeGetOperationFunction):
    def _make_get_op_fn(
            self,
            input,
            get_cost_fn,
            get_opt_fn
            ):
        return MakeGetOperationFunction._make_get_op_fn(input, self, get_cost_fn, get_opt_fn)

class Multi_Machine_Trainer(Batch_Expand_Multi_GPU_Trainer, MakeGetOperationFunction):
    def _make_get_op_fn(
            self,
            input,
            get_cost_fn,
            get_opt_fn
            ):
        return MakeGetOperationFunction._make_get_op_fn(input, self, get_cost_fn, get_opt_fn)



def duration_to_alignment(duration, max_size = None, text_mask = None, sigma = 10.):
    if not exists(max_size):
        max_size = tf.reduce_sum(duration, axis = -1)
        max_size = tf.reduce_max(max_size)
        max_size = tf.maximum(1., tf.ceil(max_size))

    token_size = duration
    token_ends = tf.math.cumsum(token_size, axis = 1)
    token_centers = (token_ends - (token_size / 2.))

    out_position = tf.range(max_size, dtype = tf.float32)[None]
    difference = token_centers[:, None] - out_position[:, :, None]
    logits = - (difference ** 2 / sigma)

    if exists(text_mask):
        logits *= text_mask[:, None]
        logits -= 1e6 * (1. - text_mask)[:, None]

    align = tf.nn.softmax(logits)
    align = tf.reshape(align, [tf.shape(duration)[0], -1, tf.shape(duration)[1]])

    return tf.clip_by_value(align, 0., 1.)



TRN = hp.train

class Solver(ModelDesc):
    def __init__(
            self,
            sampling_rate,
            window_size,
            hop_size,
            fft_size,
            f_size,
            f_max,
            f_min,
            log_scaler,
            **optim_kwargs
            ):
        self.SR = sampling_rate
        self.W = window_size
        self.H = hop_size
        self.FFT = fft_size
        self.D_F = f_size
        self.M_F = f_max
        self.m_F = f_min
        self.log_scaler = log_scaler
        self.optim_kwargs = optim_kwargs
        self.global_step = tf.train.get_or_create_global_step()

    def get_optimizer(self):
        keys = ('initial_learning_rate', 'terminal_learning_rate', 'decay_steps', 'decay_power', 'decay_cycle', 'weight_decay')
        key_checks = list(map(lambda key: key in self.optim_kwargs.keys(), keys))
        init_lr_key = keys[0] if key_checks[0] else 'learning_rate'
        term_lr_key = keys[1] if key_checks[1] else init_lr_key
        decay = key_checks[2] and init_lr_key != term_lr_key

        if decay:
            learning_rate = tf.train.polynomial_decay(
                    learning_rate = self.optim_kwargs[init_lr_key],
                    global_step = self.global_step,
                    decay_steps = self.optim_kwargs[keys[2]],
                    end_learning_rate = self.optim_kwargs[term_lr_key],
                    power = self.optim_kwargs[keys[3]] if key_checks[3] else 1.0,
                    cycle = self.optim_kwargs[keys[4]] if key_checks[4] else False,
                    name = 'learning_rate_decaying'
                    )

        else:
            learning_rate = self.optim_kwargs[init_lr_key]

        with tf.name_scope('train'): tf.summary.scalar('learning_rate', learning_rate)

        optim_kwargs = {
                'learning_rate': learning_rate,
                'beta1': self.optim_kwargs['beta_1'],
                'beta2': self.optim_kwargs['beta_2'],
                'epsilon': self.optim_kwargs['epsilon'],
                }

        if key_checks[5] and self.optim_kwargs[keys[5]] > 0:
            optimizer = AdamWeightDecayOptimizer(weight_decay_rate = self.optim_kwargs[keys[5]], **optim_kwargs)
        else:
            optimizer = tf.train.AdamOptimizer(**optim_kwargs)

        return optimizer


    def wave_to_images(
            self,
            wave,
            sampling_rate = None,
            window_size = None,
            hop_size = None,
            fft_size = None,
            mel_size = None,
            min_frequency = None,
            max_frequency = None,
            log_scaler = None,
            ):
        if sampling_rate is None: sampling_rate = self.SR
        if window_size is None: window_size = self.W
        if hop_size is None: hop_size = self.H
        if fft_size is None: fft_size = self.FFT
        if mel_size is None: mel_size = self.D_F
        if min_frequency is None: min_frequency = self.m_F
        if max_frequency is None: max_frequency = self.M_F
        if log_scaler is None: log_scaler = self.log_scaler

        wave_size, pad_size = shape(wave)[1], int(window_size - hop_size) // 2
        wave = tf.pad(wave, ((0, 0), (pad_size, -wave_size % hop_size + pad_size)), mode='REFLECT')
        transform = tf.signal.stft(
                signals = wave,
                frame_length = window_size,
                frame_step = hop_size,
                fft_length = fft_size,
                )
        transform /= (fft_size / 1920)

        #spectrogram = tf.sqrt(tf.pow(tf.real(transform), 2) + tf.pow(tf.imag(transform), 2) + 1e-9)
        spectrogram = tf.abs(transform)

        # 1. unscaled, i.e. magnitude
        unscaled_spectrogram = spectrogram / log_scaler
        unscaled_spectrogram = tf.where(unscaled_spectrogram > 1., 1. + tf.log(unscaled_spectrogram), unscaled_spectrogram)

        # 2. scaled, i.e. mel
        translate_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins = mel_size,
                num_spectrogram_bins = fft_size // 2 + 1,
                sample_rate = sampling_rate,
                lower_edge_hertz = min_frequency,
                upper_edge_hertz = max_frequency,
                dtype = tf.float32,
                )
        scaled_spectrogram = tf.einsum('bls,sm->blm', spectrogram, translate_matrix) * (mel_size / 120) / log_scaler
        scaled_spectrogram = tf.where(scaled_spectrogram > 1., 1. + tf.log(scaled_spectrogram), scaled_spectrogram)

        # 3. energy
        #energy = tf.norm(spectrogram, axis = -1) / log_scaler
        #energy = tf.where(energy > 1., 1. + tf.log(energy), energy)
        #energy = tf.minimum(energy, 14.)

        return unscaled_spectrogram, scaled_spectrogram


    def inputs(self):
        inputs = (
                #(tf.string, [None], 'path'),
                (tf.int32, [None], 'speaker_id'),
                (tf.int32, [None, None], 'tokens'),
                (tf.int32, [None], 'token_size'),
                (tf.float32, [None, None], 'wave'),
                (tf.int32, [None], 'wave_size'),
                #(tf.float32, [None, None], 'pitch'),
                )
        return list(map(tf.placeholder, *zip(*inputs)))



ALGN = hp.aligner

class Aligner_Solver(Solver):
    def __init__(
            self,
            sampling_rate,
            window_size,
            hop_size,
            fft_size,
            f_size,
            f_max,
            f_min,
            log_scaler,
            shift,
            save_modules,
            **optim_kwargs
            ):
        super(Aligner_Solver, self).__init__(sampling_rate, window_size, hop_size, fft_size, f_size, f_max, f_min, log_scaler, **optim_kwargs)

        self.shift = shift
        self.save_modules = save_modules


    def build_graph(
            self,
            speaker_id,
            text_id,
            text_size,
            wave,
            wave_size,
            ):

        mel = self.wave_to_images(wave = wave)[1]
        #with tf.control_dependencies([tf_print('mel', mel, color='cyan')]): mel = tf.identity(mel)

        image_size = (- wave_size % self.H + wave_size) // self.H
        image_mask = tf.sequence_mask(image_size, shape(mel)[1], dtype = tf.float32)
        image_pad_mask = tf.pad(image_mask, ((0, 0), (self.shift, 0)), constant_values = 1)

        text_mask = tf.sequence_mask(text_size, shape(text_id)[1], dtype = tf.float32)

        mel_pred, align_map = aligner(
                text = text_id,
                image = tf.pad(mel, ((0, 0), (self.shift, 0), (0, 0))), # 혹시 mel최솟값이 달라지면 해당 최솟값으로
                speaker = speaker_id,
                text_capacity = len(tokenizer.all_symbols),
                text_mask = text_mask,
                image_mask = tf.tile(image_pad_mask[Ellipsis, None], (1, 1, self.D_F)),
                trainable = True,
                **ALGN
                )
        align_pred = tf.nn.softmax(align_map[:, :-self.shift])

        align_mask = text_mask[:, None] * image_mask[Ellipsis, None]
        align_pad_mask = text_mask[:, None] * image_pad_mask[Ellipsis, None]
        D = - tf.nn.log_softmax(align_map) * align_pad_mask
        R = softdtw_tfscan(D, gamma = 0.1)
        log_align_logit = tf.py_func(compute_softdtw_backward, (D, R, image_size, text_size, 0.1), tf.float32)
        align_logit = tf.where(tf.is_nan(log_align_logit), tf.zeros_like(log_align_logit), tf.exp(log_align_logit))
        align_gt = align_logit / tf.maximum(tf.reduce_sum(align_logit, axis = 2, keepdims = True), 1e-31)
        align_gt = align_gt * align_pad_mask
        dur_gt = tf.reduce_sum(align_gt, axis = 1)

        mel_preds = tf.split(mel_pred, self.shift, axis = -1)
        loss_image, gt_images, pred_images = [], [], []

        b, l_plus_shift, d = shape(mel_preds[0])
        l = shape(mel)[1]
        for n in range(self.shift):
            shift_pred = tf.slice(mel_preds[n], begin = (0, n, 0), size = [b, l_plus_shift - n, d])
            pred_images.append(shift_pred)
            shift = tf.math.abs(tf.pad(mel, ((0, 0), (0, l_plus_shift - n - l), (0, 0))) - shift_pred)
            shift = tf.reduce_mean(shift, axis = 2)
            shift *= tf.pad(image_mask, ((0, 0), (0, l_plus_shift - n - l)))
            shift = tf.reduce_sum(shift, axis = 1)
            shift /= tf.cast(l_plus_shift - n, shift.dtype)
            loss_image.append(shift) #TODO image가 쉬프트된 크기가 클수록 맞추기 어려우므로, 보정 상수를 곱해준다.
        loss_image = tf.reduce_mean(loss_image)

        aligner_ = {}
        aligner_['image'] = TRN.loss_multiplier.aligner.image * loss_image

        aligner_['total'] = tf.reduce_sum(list(aligner_.values()))

        local = locals()
        with tf.name_scope('loss'):
            for key in aligner_.keys():
                local['loss_ALGN_' + key] = aligner_[key]
                tf.add_to_collection(tf.GraphKeys.LOSSES, local['loss_ALGN_' + key])
                tf.summary.scalar('loss_ALGN_' + key, aligner_[key])

        for module in self.save_modules:
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, f'^{module}'):
                tf.add_to_collection(module, variable)

        with tf.name_scope('visual'):
            align_rc = duration_to_alignment(dur_gt[:1], shape(mel)[1], text_mask[:1], 10.0) * align_mask[:1]

            def align_norm(align):
                m, M = tf.reduce_min(align), tf.reduce_max(align)
                return (align - m) / (M - m)

            align_rc_norm = align_norm(align_rc[:1])
            align_rc_norm = tf.transpose(align_rc_norm, [0, 2, 1])

            align_rc_image = [
                    align_rc_norm[:, ::-1],
                    tf.zeros_like(align_rc_norm),
                    tf.zeros_like(align_rc_norm),
                    tf.ones_like(align_rc_norm),
                    ]
            align_rc_image = tf.stack(align_rc_image, axis = 3)

            align_pred_norm = align_norm(align_pred[:1])
            align_pred_norm = tf.transpose(align_pred_norm, [0, 2, 1])

            align_pred_image = [
                    tf.zeros_like(align_pred_norm),
                    align_pred_norm[:, ::-1],
                    tf.zeros_like(align_pred_norm),
                    tf.ones_like(align_pred_norm),
                    ]
            align_pred_image = tf.stack(align_pred_image, axis = 3)

            tf.summary.image('aligner_mel_gt', colorize(mel[:1], 0., 12.), 1)
            tf.summary.image('aligner_mel_pred', colorize(mel_pred[:1], 0., 12.), 1)
            for n in range(self.shift):
                tf.summary.image(f'aligner_mel_pred_shift{n}', colorize(pred_images[n][:1], 0., 12.), 1)

            tf.summary.image('aligner_align_compare', align_pred_image + align_rc_image, 1)

            tf.summary.audio('aligner_wave', wave[:1], self.SR, 1)

        return  (
                    [
                        local['loss_ALGN_total'],
                    ],
                    [
                        [
                            ALGN.scope,
                        ],
                    ]
                )


DUR = hp.duration

class Duration_Solver(Solver):
    def __init__(
            self,
            sampling_rate,
            window_size,
            hop_size,
            hop_size_align,
            fft_size,
            f_size,
            f_max,
            f_max_align,
            f_min,
            f_min_align,
            log_scaler,
            shift,
            save_modules,
            **optim_kwargs
            ):
        super(Duration_Solver, self).__init__(sampling_rate, window_size, hop_size, fft_size, f_size, f_max, f_min, log_scaler, **optim_kwargs)

        self.H_A = hop_size_align
        self.m_F_A = f_min_align
        self.M_F_A = f_max_align
        self.shift = shift
        self.save_modules = save_modules


    def build_graph(
            self,
            speaker_id,
            text_id,
            text_size,
            wave,
            wave_size,
            ):

        mel_dur = self.wave_to_images(wave = wave)[1]
        mel_align = self.wave_to_images(
                wave = wave,
                hop_size = self.H_A,
                min_frequency = self.m_F_A,
                max_frequency = self.M_F_A,
                )[1]

        image_size_align = (- wave_size % self.H_A + wave_size) // self.H_A
        image_size_dur = (- wave_size % self.H + wave_size) // self.H

        image_mask_align = tf.sequence_mask(image_size_align, shape(mel_align)[1], dtype = tf.float32)
        image_pad_mask_align = tf.pad(image_mask_align, ((0, 0), (self.shift, 0)), constant_values = 1)

        image_mask_dur = tf.sequence_mask(image_size_dur, shape(mel_dur)[1], dtype = tf.float32)

        text_mask = tf.sequence_mask(text_size, shape(text_id)[1], dtype = tf.float32)

        _, align_map = aligner(
                text = text_id,
                image = tf.pad(mel_align, ((0, 0), (self.shift, 0), (0, 0))), # 혹시 mel최솟값이 달라지면 해당 최솟값으로
                speaker = speaker_id,
                text_capacity = len(tokenizer.all_symbols),
                text_mask = text_mask,
                image_mask = tf.tile(image_pad_mask_align[Ellipsis, None], (1, 1, self.D_F)),
                trainable = False,
                **ALGN
                )
        align_pred = tf.nn.softmax(align_map[:, :-self.shift])

        align_mask = text_mask[:, None] * image_mask_dur[Ellipsis, None]
        align_pad_mask = text_mask[:, None] * image_pad_mask_align[Ellipsis, None]
        D = - tf.nn.log_softmax(align_map) * align_pad_mask
        R = softdtw_tfscan(D, gamma = 0.1)
        log_align_logit = tf.py_func(compute_softdtw_backward, (D, R, image_size_align, text_size, 0.1), tf.float32)
        align_logit = tf.where(tf.is_nan(log_align_logit), tf.zeros_like(log_align_logit), tf.exp(log_align_logit))
        align_gt = align_logit / tf.maximum(tf.reduce_sum(align_logit, axis = 2, keepdims = True), 1e-31)
        align_gt = align_gt * align_pad_mask
        dur_gt = tf.reduce_sum(align_gt, axis = 1)

        dur_gt *= self.H_A / self.H

        dur_pred = duration_allocator(
                text = text_id,
                speaker = speaker_id,
                text_capacity = len(tokenizer.all_symbols),
                text_mask = text_mask,
                trainable = True,
                **DUR
                )

        # loss
        loss_l1 = tf.reduce_sum(tf.abs(dur_pred - tf.stop_gradient(dur_gt)), axis = 1)
        loss_l1 /= tf.cast(text_size, tf.float32)
        loss_l1 = tf.reduce_mean(loss_l1)

        duration_ = {}
        duration_['l1'] = TRN.loss_multiplier.duration.l1 * loss_l1

        duration_['total'] = tf.reduce_sum(list(duration_.values()))

        local = locals()
        with tf.name_scope('loss'):
            for key in duration_.keys():
                local['loss_DUR_' + key] = duration_[key]
                tf.add_to_collection(tf.GraphKeys.LOSSES, local['loss_DUR_' + key])
                tf.summary.scalar('loss_DUR_' + key, duration_[key])

        for module in self.save_modules:
            for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, f'^{module}'):
                tf.add_to_collection(module, variable)

        with tf.name_scope('visual'):
            am_gt = tf.cast(tf.math.round(tf.cumsum(dur_gt[:1], 1)), tf.int32)
            am_gt = tf.reduce_max(tf.one_hot(am_gt, shape(mel_dur)[1]), axis = 1)[:, None, :, None]

            ad_gt = [
                    tf.zeros_like(am_gt),
                    tf.zeros_like(am_gt),
                    am_gt,
                    tf.ones_like(am_gt)
                    ]
            ad_gt = tf.concat(ad_gt, axis = 3)


            align_rc = duration_to_alignment(dur_gt[:1], shape(mel_dur)[1], text_mask[:1], 10.0) * align_mask[:1]
            align_rc = tf.transpose(align_rc, [0, 2, 1])

            align_rc_image = [
                    align_rc[:, ::-1],
                    tf.zeros_like(align_rc),
                    tf.zeros_like(align_rc),
                    tf.ones_like(align_rc),
                    ]
            align_rc_image = tf.stack(align_rc_image, axis = 3)


            align_pred = duration_to_alignment(tf.nn.relu(dur_pred[:1]), shape(mel_dur)[1], text_mask[:1], 10.0) * align_mask[:1]
            align_pred = tf.transpose(align_pred, [0, 2, 1])

            align_pred_image = [
                    tf.zeros_like(align_pred),
                    align_pred[:, ::-1],
                    tf.zeros_like(align_pred),
                    tf.ones_like(align_pred),
                    ]
            align_pred_image = tf.stack(align_pred_image, axis = 3)

            tf.summary.image('aligner_mel_gt', colorize(mel_align[:1], 0., 12.), 1)
            tf.summary.image('duration_mel_gt', colorize(mel_dur[:1], 0., 12.) * (1. - am_gt) + ad_gt * am_gt, 1)
            tf.summary.image('duration_align_compare', align_pred_image + align_rc_image, 1)
            tf.summary.audio('duration_wave', wave[:1], self.SR, 1)

        return  (
                    [
                        local['loss_DUR_total'],
                    ],
                    [
                        [
                            DUR.scope,
                        ],
                    ]
                )



TF = hp.transformer

class Transformer_Solver(Solver):
    def __init__(
            self,
            sampling_rate,
            window_size,
            hop_size,
            fft_size,
            f_size,
            f_max,
            f_min,
            log_scaler,
            shift,
            save_modules,
            **optim_kwargs
            ):
        super(Transformer_Solver, self).__init__(sampling_rate, window_size, hop_size, fft_size, f_size, f_max, f_min, log_scaler, **optim_kwargs)

        self.save_modules = save_modules


    def build_graph(
            self,
            speaker_id,
            text_id,
            text_size,
            wave,
            wave_size,
            ):
        mel = self.wave_to_images(wave)[1]

        image_size = (- wave_size % self.H + wave_size) // self.H
        image_mask = tf.sequence_mask(image_size, shape(mel)[1], dtype = tf.float32)

        text_mask = tf.sequence_mask(text_size, shape(text_id)[1], dtype = tf.float32)

        duration = duration_allocator(
                text = text_id,
                speaker = speaker_id,
                text_capacity = len(tokenizer.all_symbols),
                text_mask = text_mask,
                trainable = True,
                **DUR
                )

        mel_input = tf.pad(mel, ((0, 0), (1, 0), (0, 0)))[:, :-1]
        mel_output = causal_speech_model(
                text_id,
                duration,
                speaker_id,
                mel_input,
                text_mask = text_mask,
                image_mask = image_mask,
                trainable = True,
                **TF
                )

        # mel_output vs mel_gt







def train(case, gpu, mode, speakers = ''):

    num_gpu = len(gpu)
    if num_gpu > 1:
        cprint('parallel-machine training', color = 'red')
        trainer = Multi_Machine_Trainer(gpus = num_gpu, penalty_targets = TRN.penalty_targets, penalty_rate = TRN.penalty_rate)
    else:
        cprint('single-machine training', color = 'red')
        trainer = Single_Machine_Trainer(penalty_targets = TRN.penalty_targets, penalty_rate = TRN.penalty_rate)

    if mode == 'aligner':
        save_modules = [ALGN.scope]
        restore_modules = [f'{mode}/' + module for module in save_modules]

        solver = Aligner_Solver(
                sampling_rate = AUD.sampling_rate,
                window_size = AUD.window_size,
                hop_size = AUD.aligner_hop_size,
                fft_size = AUD.fft_size,
                f_size = AUD.mel_frequency,
                f_max = AUD.max_frequency,
                f_min = AUD.min_frequency,
                log_scaler = AUD.log_scaler,
                shift = ALGN.shift,
                save_modules = save_modules,
                **hp.optimizer
                )

    if mode == 'duration':
        save_modules = [f'{DUR.scope}/cond', f'{DUR.scope}/tknz', f'{DUR.scope}/allc', f'{DUR.scope}/conv']
        restore_modules = [f'aligner/{ALGN.scope}']
        restore_modules += [f'{mode}/' + module for module in save_modules]

        solver = Duration_Solver(
                sampling_rate = AUD.sampling_rate,
                window_size = AUD.window_size,
                hop_size = AUD.hop_size,
                hop_size_align = AUD.aligner_hop_size,
                fft_size = AUD.fft_size,
                f_size = AUD.mel_frequency,
                f_max = AUD.max_frequency,
                f_max_align = AUD.aligner_max_frequency,
                f_min = AUD.min_frequency,
                f_min_align = AUD.aligner_min_frequency,
                log_scaler = AUD.log_scaler,
                shift = ALGN.shift,
                save_modules = save_modules,
                **hp.optimizer
                )

    if mode == 'transformer':
        save_modules = [f'{TF.scope}/cond', f'{TF.scope}/encoder', f'{TF.scope}/decoder']
        restore_modules = [f'duration/{DUR.scope}/' + module for module in ['cond', 'tknz', 'allc', 'conv']]
        restore_module += [f'{mode}/' + module for module in save_modules]

    dataset = Dataset(
            data_path = TRN.data_path,
            gpus = num_gpu,
            package_size = TRN.package_size,
            model_size = ALGN.features,
            speakers = speakers,
            clusters = TRN.clusters,
            slicer = TRN.slicer,
            sampling_rate = AUD.sampling_rate,
            hop_size = AUD.hop_size,
            voice_left = TRN.voice.left,
            voice_right = TRN.voice.right,
            dBFS = AUD.dBFS,
            )

    train_dataflow, train_batch = dataset(
            threads = TRN.threads,
            buffers = TRN.buffers,
            )


    config = tf.ConfigProto()

    checkpoint_path = osp.join(TRN.log_path, case)
    logger_path = osp.join(checkpoint_path, mode)
    logger.set_logger_dir(logger_path, action = 'k' if exists(TRN.transfer_path) else None)

    monitors = [TFEventWriter(), JSONWriter(), ScalarPrinter()]

    restore_path = osp.join(TRN.log_path, TRN.transfer_path) if exists(TRN.transfer_path) else checkpoint_path

    restore_inits = []
    for module in restore_modules:
        restore_checkpoint = tf.train.latest_checkpoint(osp.join(restore_path, module))
        if exists(restore_checkpoint):
            restore_init = SaverRestore(
                    model_path = restore_checkpoint,
                    prefix = None,
                    ignore = ['global_step'] if module.split(osp.sep)[0] != mode else []
                    )
            restore_inits.append(restore_init)

    session_init = ChainInit(restore_inits) if len(restore_inits) > 0 else None

    callbacks = [
        ModelSaver(
            max_to_keep = 5,
            keep_checkpoint_every_n_hours = 1,
            checkpoint_dir = osp.join(checkpoint_path, mode, module),
            var_collections = [tf.GraphKeys.GLOBAL_STEP, module],
            ) for module in save_modules
        ]

    epoch = JSONWriter.load_existing_epoch_number()

    train_configuration = TrainConfig(
        model = solver,
        dataflow = train_dataflow,
        callbacks = callbacks,
        steps_per_epoch = TRN.steps_per_epoch,
        session_init = session_init,
        starting_epoch = 1 + (epoch if epoch else 0),
        monitors = monitors,
        max_epoch = 99999,
        )

    launch_train_with_config(train_configuration, trainer = trainer)

