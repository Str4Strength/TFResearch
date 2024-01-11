import os
import re
import math
import random
import itertools

import resampy
import torch
import torchcrepe

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import soundfile as sf

from tqdm import tqdm
from pydub import AudioSegment
from librosa.filters import mel as librosa_mel_fn
from termcolor import colored, cprint
from multiprocessing import Process, Queue, Manager


from tensorpack.dataflow import (
        MultiProcessMapAndBatchDataZMQ, DataFromList, BatchData, RepeatedData)

from tensorpack.train.trainers import (
        SimpleTrainer, SyncMultiGPUTrainerReplicated, _int_to_range)

from tensorpack.input_source import FeedfreeInput

from tensorpack.graph_builder import DataParallelBuilder
from tensorpack.graph_builder.training import SyncMultiGPUReplicatedBuilder
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.graph_builder.utils import (
        GradientPacker, merge_grad_list, split_grad_list, allreduce_grads,
        allreduce_grads_hierarchical, allreduce_grads_naive,
        override_to_local_variable)

from tensorpack.callbacks import RunOp

from tensorpack.utils import logger

from tensorpack.tfutils.tower import TrainTowerContext, TowerFunc, get_current_tower_context
from tensorpack.tfutils.gradproc import FilterNoneGrad

from ..Neural_Network import *



def tf_print(name, value, summary=True, color='yellow'):
    return tf.print(
            colored("{}:".format(name), color),
            tf.shape(value),
            tf.reduce_min(value), '~',
            tf.reduce_mean(value), '~',
            tf.reduce_max(value),
            "\n" if not summary else '',
            value if not summary else '',
            summarize=256)



# meta-class for Dataset
class Dataset():
    def __init__(
            self,
            data_path,
            speakers,
            batch_size,
            sampling_rate,
            window_size,
            hop_size,
            fft_size,
            meltime_size,
            melfreq_size,
            ):
        # dataset packaging
        self.data_path = data_path
        self.speakers = speakers
        self.labels = [vec for vec in np.eye(len(speakers))]
        #cprint(self.labels, 'magenta')
        #cprint((len(speakers), len(self.labels)), 'blue')
        self.spk_to_lb = dict(zip(speakers, self.labels))
        #self.lb_to_spk = dict(zip(self.labels, speakers))
        self.batch_size = batch_size
        # audio process
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.meltime_size = meltime_size
        self.melfreq_size = melfreq_size


    def _get_array_(
            self,
            file_path,
            array_size=None,
            array_begin=None,
            signal_size=None,
            signal_begin=None,
            dtype='float32'):
        if isinstance(signal_size, int) and not isinstance(array_size, int):
            array_size = signal_size // self.hop_size #+1
        if isinstance(signal_begin, int) and not isinstance(array_begin, int):
            #array_begin = (signal_beigin - (self.hop_size // 2)) // self.hop_size
            array_begin = signal_beigin // self.hop_size

        array = np.load(file_path, mmap_mode='r').astype(dtype)
        raw_length = len(array)
        array_ndims = len(array.shape)

        if isinstance(array_size, int):
            if isinstance(array_begin, int):
                real_length = min(array_size, raw_length - array_begin)
                if real_length < 0: raise ValueError
                array = array[array_begin: array_begin + real_length]
            elif raw_length > array_size:
                real_length = array_size
                array_begin = random.randint(0, raw_length - array_size)
                array = array[array_begin: array_begin + real_length]
            else:
                real_length = len(array)
                array_begin = 0

            if real_length < array_size:
                padding = (0, array_size - real_length)
                if array_ndims > 1: padding = tuple([padding] + [(0, 0)]*(array_ndims-1))
                array = np.pad(array, padding, 'constant')
            assert len(array) == array_size

        elif isinstance(array_begin, int):
            array = array[array_begin:]
            real_length = len(array)

        return array, real_length, raw_length, array_begin, signal_begin


    def read_normalized(
            self,
            file_,
            target_dBFS=-23.0
            ):
        aud_obj = AudioSegment.from_file(file_)
        norm_obj = aud_obj.apply_gain(target_dBFS - aud_obj.dBFS)
        sound = np.array(norm_obj.get_array_of_samples())/2**15
        return sound


    def _get_signal_(
            self,
            file_path,
            signal_size=None,
            signal_begin=None,
            array_size=None,
            array_begin=None,
            dtype='float32',
            normalize_volume=True,
            norm_dBFS=-23.0
            ):
        if isinstance(array_size, int) and not isinstance(signal_size, int):
            signal_size = array_size * self.hop_size
        if isinstance(array_begin, int) and not isinstance(signal_begin, int):
            signal_begin = (array_begin * self.hop_size) #+ (self.hop_size // 2)

        ### TODO only for 16bit, header size 44 !!!
        raw_length = (os.stat(file_path).st_size - 44) // 2
        real_length = raw_length

        if isinstance(signal_size, int):
            if isinstance(signal_begin, int):
                real_length = min(signal_size, raw_length - signal_begin)
                if real_length < 0: cprint("real_length_negative", color='red')
                signal = sf.read(file_path, frames=real_length, start=signal_begin, dtype=dtype)[0]
                case_ = 1
            elif raw_length > signal_size:
                real_length = signal_size
                signal_begin = random.randint(0, raw_length - signal_size)
                signal = sf.read(file_path, frames=signal_size, start=signal_begin, dtype=dtype)[0]
                case_ = 2
            else:
                signal = sf.read(file_path, dtype=dtype)[0]
                real_length = len(signal)
                case_ = 3

            if real_length < signal_size:
                signal = np.pad(signal, (0, signal_size - real_length), 'linear_ramp')
            assert len(signal) == signal_size

        elif isinstance(signal_begin, int):
            signal = sf.read(file_path, start=signal_begin, dtype=dtype)[0]
            real_length = len(signal)

        else:
            signal = sf.read(file_path, dtype=dtype)[0]
            real_length = len(signal)

        if normalize_volume:
            signal *= (10**(norm_dBFS/20) / max(1e-9, np.sqrt(np.mean(np.square(signal)))))

        return signal, real_length, raw_length, signal_begin, array_begin


    def _f0_torchcrepe(self, signal, threshold=0.05, device="cpu"):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        sig16k = resampy.resample(signal, self.sampling_rate, 16000)
        sig16k_torch = torch.FloatTensor(sig16k).unsqueeze(0).to(device)

        f0, pd = torchcrepe.predict(
                sig16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=1024,
                device=device, return_periodicity=True)

        pd = torchcrepe.filter.median(pd, 3)
        pd = torchcrepe.threshold.Silence(-60.)(pd, sig16k_torch, 16000, 80)
        f0 = torchcrepe.threshold.At(threshold)(f0, pd)
        f0 = torchcrepe.filter.mean(f0, 3)

        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

        nzindex = torch.nonzero(f0[0]).squeeze()
        f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
        time_org = 0.005 * nzindex.cpu().numpy()
        L = signal.shape[0]

        time_frame = np.arange((-L%self.hop_size + L // self.hop_size)) * self.hop_size / self.sampling_rate

        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])

        return f0



    def __call__(
            self,
            mode,
            data_list,
            split_ratio=.2,
            n_prefetch=64,
            n_process=96,
            shuffle=True,
            scope='dataset'):
        total_length = len(data_list)
        split_ratio = max(split_ratio, 1.-split_ratio)
        cut_index = round(total_length * split_ratio)

        if mode == 'train':
            data_ = data_list[:cut_index]
        elif mode == 'evaluate':
            data_ = data_list[cut_index:]
        else:
            raise ValueError

        with tf.variable_scope(scope):
            dataset = DataFromList(data_, shuffle=shuffle)
            dataset = MultiProcessMapAndBatchDataZMQ(
                    dataset, num_proc=n_process, map_func=self.get_input,
                    batch_size=self.batch_size, buffer_size=n_prefetch)
        return dataset, total_length, len(data_)




# meta-class for Trainers and Graph Builders
#drop = lambda x: tf.where(
#        tf.logical_or(tf.math.is_inf(x), tf.math.is_nan(x)),
#        tf.zeros_like(x),
#        x)
drop = lambda x: tf.where(
        tf.logical_or(tf.logical_or(tf.math.greater(x, 50.0), tf.math.less(x, -50.0)), tf.math.is_nan(x)),
        tf.zeros_like(x),
        x)

nan_inf = lambda p: (lambda g, v: (drop(g), v))(*p)


class MakeGetOperationFunction():
    @staticmethod
    def _make_get_op_fn(input, class_, get_cost_fn, get_opt_fn):
        assert input.setup_done()

        def get_op_fn():
            context, inputs = [get_current_tower_context(), input.get_input_tensors()]
            costs, train_modules = get_cost_fn(*inputs)
            for cost in costs: assert isinstance(cost, tf.Tensor) and cost.shape.ndims == 0

            get_collection = (
                    context.get_collection_in_tower
                    if context.has_own_variables
                    else tf.get_collection
                    )
            train_vars, move_vars = list(map(lambda key_vars: get_collection(key_vars),
                (tf.GraphKeys.TRAINABLE_VARIABLES, "MOVING_AVERAGE")))

            opt = get_opt_fn()
            common_kwargs = {
                    "gate_gradients": class_.GATE_GRADIENTS,
                    "colocate_gradients_with_ops": class_.COLOCATE_GRADIENTS_WITH_OPS,
                    "aggregation_method": class_.AGGREGATION_METHOD
                    }

            def var_filter(vs, f): return [v for v in vs if f in v.name]
            def pair(vs): return [(v, v) for v in vs]

            gradients, movings = [], []
            for n, modules in enumerate(train_modules):
                vars_ = []
                for module_name in modules:
                    print(module_name)
                    vars_ += var_filter(train_vars, module_name)

                grads_ = opt.compute_gradients(costs[n], var_list = vars_, **common_kwargs)
                grads_ = FilterNoneGrad().process(grads_)
                gradients.append(grads_)

                moves = pair(vars_)
                movings.append(moves)

            return (gradients, movings,)

        return get_op_fn



# Training via Single GPU
# covers all kinds of module numbers
class Single_GPU_Trainer(SimpleTrainer):
    def __init__(
            self,
            penalty_targets=None,
            penalty_rate=1.0
            ):
        super().__init__()
        self.penalty_targets = penalty_targets
        self.penalty_rate = penalty_rate

    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        logger.info("Building graph for a single training tower ...")

        with TrainTowerContext(""):
            grads, moves = self._make_get_op_fn(input, get_cost_fn, get_opt_fn)()
            # n_flow, n_vars, 2
            #cprint(len(grads[0]), "cyan")

            tags = [f"train_{n}" for n in range(len(grads)-1)] + ["train_op"]
            #cprint(tags, "cyan")

            opt = get_opt_fn()

            def apply_grads_to_vars(gv, penalty_target=None, penalty_ratio=1.0,
                    name="name"):
                naninf = list(map(nan_inf, gv))
                #cprint(naninf, 'cyan')
                if penalty_target is None:
                    gv = naninf
                else:
                    gv = []
                    for g, v in naninf:
                        if penalty_target in v.name: g *= penalty_ratio
                        gv.append((g, v))
                return [opt.apply_gradients(gv, name=f"apply_grads_{name}")\
                       if len(gv) > 0 else tf.constant(0.)]

            def assign_moves_to_vars(mvs, name="name"):
                naninf = list(map(nan_inf, mv))
                return [state_ops.assign(m, v) for m, v in naninf]

            with tf.name_scope("apply_gradients"):
                train_flow = []
                for n, tag in enumerate(tags):
                    #print(train_flow)
                    #print(grads[n])
                    with tf.control_dependencies(train_flow):
                        train_flow = apply_grads_to_vars(
                                grads[n],
                                self.penalty_targets[n],
                                self.penalty_rate[n],
                                tag)
                        """
                        try:
                            train_flow += assign_moves_to_vars(
                                    moves[n],
                                    tag)
                        except:
                            pass
                        """
                #cprint(train_flow, 'cyan')
                self.train_op = train_flow
        return []

    def _make_get_op_fn(self, input, get_cost_fn, get_opt_fn):
        def get_op_fn():
            return []

        return get_op_fn


# Training via Multi GPU
# covers all kinds of module numbers
class Batch_Expand_Multi_GPU_Graph_Builder(SyncMultiGPUReplicatedBuilder):
    def __init__(
            self,
            gpus,
            average=True,
            mode=None,
            penalty_targets=None,
            penalty_rate=0.
            ):
        super().__init__(gpus, average, mode)
        self.penalty_targets = penalty_targets
        self.penalty_rate = penalty_rate

    def build(self, operations, get_opt_fn):
        assert len(operations) == len(self.towers)
        devices = ["/gpu:{}".format(k) for k in self.towers]

        # N_towers, 2, N_flows, ... --> 2 x N_towers, N_flows, ...
        grads_vars, moves_vars = list(zip(*operations))
        # Each N_flows, N_towers, N_vars, 2 (value - gradient/update, variables)
        gvs_list = list(zip(*grads_vars))
        mvs_list = list(zip(*moves_vars))

        def _check_nccl_mode(op_var_list):
            if self._mode != "nccl": return
            SyncMultiGPUReplicatedBuilder._check_grad_list(op_var_list)
            dtypes_nccl = [tf.float32, tf.float64]
            if tuple(map(int, tf.__version__.split(".")[:2])) >= (1, 8):
                dtypes_nccl.append(tf.float16)
            if not all(k in dtypes_nccl
                    for k in {x[0].dtype.base_dtype for x in op_var_list[0]}):
                logger.warn(
                "Cant use mode=`nccl` because some operations have"
                " unsupported types. Fallback to mode=`cpu`")
                sekf._mode = "cpu"

        map(lambda grad_flow: _check_nccl_mode(grad_flow), gvs_list)

        def to_main_branch(ops_list, devices):
            if self._mode in ["nccl", "collective"]:
                reduced = allreduce_grads(
                        ops_list, average=self._average, mode=self._mode)
            elif self._mode == "hierarchical":
                reduced = allreduce_grads_hierarchical(
                        ops_list, devices, average=self._average)
            else:
                devices = ["/cpu:0"] if self._mode == "cpu" else devices
                reduced = allreduce_grads_naive(
                        ops_list, devices, average=self._average)
                reduced = [reduced] * len(self.towers)
            return reduced

        def adjust(ovs_list, devices):
            if ovs_list is None: return pairs
            ops_, vars_ = split_grad_list(ovs_list)
            if self._mode in ["hierarchical"]:
                packer = GradientPacker(len(devices))
                if packer.compute_strategy(ops_[0]):
                    ops_ = packer.pack_all(ops_, devices)
                    ops_ = to_main_branch(ops_, devices)
                    ops_ = packer.unpack_all(ops_, devices)
            else:
                ops_ = to_main_branch(ops_, devices)
            ovs_list = merge_grad_list(ops_, vars_)
            return ovs_list

        grads_vars_list = [adjust(gvs, devices) for gvs in gvs_list] # Nf,Nt,Nv,2
        moves_vars_list = [adjust(mvs, devices) for mvs in mvs_list] # Nf,Nt,Nv,2
        #cprint((
        #    len(grads_vars_list),
        #    len(grads_vars_list[0]),
        #    len(grads_vars_list[0][0])
        #    ), "magenta")
        #cprint((
        #    len(moves_vars_list),
        #    len(moves_vars_list[0]),
        #    #len(moves_vars_list[0][0])
        #    ), "cyan")
        tags = [f"train_{n}" for n in range(len(grads_vars_list)-1)]+["train_op"]
        cprint(tags, "red")
        opt = get_opt_fn()

        def apply_grads_to_vars(gvs, penalty_target=None, penalty_ratio=1.0,
                name="name"):
            # gvs : N_towers, N_variables, 2
            train_list = []
            for idx, grads_vars in enumerate(gvs): # n for N_towers
                with tf.device(devices[idx]):
                    with override_to_local_variable(enable=idx>0):
                        naninf = list(map(nan_inf, grads_vars)) # N_variables, 2
                        if penalty_target is None:
                            grads_vars = naninf
                        else:
                            grads_vars = []
                            for g, v in naninf:
                                if penalty_target in v.name: g *= penalty_ratio
                                grads_vars.append((g, v))
                        train_list.append(opt.apply_gradients(grads_vars,
                            name=f"apply_grads_{name}{idx}") if len(grads_vars)>0
                            else tf.constant(0.))
            return train_list

        def assign_moves_to_vars(mvs, name="name"):
            assign_list = []
            # mvs : N_towers, N_variables, 2
            for idx, moves_vars in enumerate(mvs): # n for N_towers
                with tf.device(devices[idx]):
                    with override_to_local_vairable(enable=idx>0):
                        assign_each_tower = []
                        for m, v in moves_vars: # for each vars
                            assign_each_tower.append(state_ops.assign(
                                m, v, name=f"assign_moves_{name}{idx}"))
                        assign_list += assign_each_tower
                        #assign_list.append(assign_each_tower)
            return assign_list

        with tf.variable_scope("apply_gradients", reuse=tf.AUTO_REUSE):
            train_flow = []
            for n, tag in enumerate(tags): # for N_process
                with tf.control_dependencies(train_flow):
                    train_flow = apply_grads_to_vars(
                            grads_vars_list[n],
                            self.penalty_targets[n],
                            self.penalty_rate[n],
                            tag)
                    try:
                        train_flow += assign_moves_to_vars(
                                moves_vars_list[n],
                                tag)
                    except:
                        train_flow = train_flow
            train_op = train_flow

        with tf.name_scope("sync_variables"):
            post_init_op = SyncMultiGPUReplicatedBuilder.get_post_init_ops()\
                    if len(self.towers) > 1 else None
        return train_op, post_init_op


class Batch_Expand_Multi_GPU_Trainer(SyncMultiGPUTrainerReplicated):
    def __init__(
            self,
            gpus,
            average=True,
            mode=None,
            penalty_targets=None,
            penalty_rate=0.):
        super().__init__(gpus, average, mode)
        gpus = _int_to_range(gpus)
        self.devices = gpus

        if mode is None:
            mode = "hierarchical" if len(gpus) == 8 else "nccl"
        mode = mode.lower()

        self._builder = Batch_Expand_Multi_GPU_Graph_Builder(
                gpus, average, mode, penalty_targets, penalty_rate)
        self.BROADCAST_EVERY_EPOCH = True

    def _setup_graph(
            self,
            input,
            get_cost_fn,
            get_opt_fn
            ):
        if len(self.devices) > 1:
            assert isinstance(input, FeedfreeInput), input
        tower_function = self._make_get_op_fn(input, get_cost_fn, get_opt_fn)
        op_list = self._builder.call_for_each_tower(tower_function)
        self.train_op, post_init_op = self._builder.build(op_list, get_opt_fn)

        if post_init_op is not None:
            cb = RunOp(
                    post_init_op,
                    run_before=True,
                    run_as_trigger=self.BROADCAST_EVERY_EPOCH,
                    verbose=True)
            cb.name_scope = "SyncVariables"
            return [cb]
        else:
            return []


class Solver(ModelDesc):
    def __init__(
            self,
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
            adam_beta_1=0.0,
            adam_beta_2=0.99,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            decay_steps=None,
            decay_alpha=None,
            ):
        self.global_step = tf.train.get_or_create_global_step()
        self.B = batch_size
        # mel-spectrogram informations
        self.T_s = meltime_size_source
        self.T_t = meltime_size_target
        self.M = melfreq_size
        # signal informations
        self.SR = sampling_rate
        self.W = window_size
        self.H = hop_size
        self.FFT = fft_size
        self.Low_Clip = low_clip_hertz
        self.High_Clip = high_clip_hertz
        #self.Log_Scale = scale_log
        # optimizer informations
        self.LR = learning_rate
        self.Dec_Steps = decay_steps
        self.Dec_Alpha = decay_alpha
        self.Adam_B1 = adam_beta_1
        self.Adam_B2 = adam_beta_2
        self.Adam_Eps = adam_epsilon
        self.W_Decay = weight_decay

    def convert_size(self, signal_size=None, meltime_size=None):
        assert signal_size is None or meltime_size is None
        if signal_size is not None:
            return (-signal_size%self.H+signal_size)//self.H
        else:
            return meltime_size * self.H

    """
    ### TODO humelo setting
    def sig_to_mag_mel(self, signal, log_scaler=0.0025):
        self.MinMel = 0.0
        with tf.variable_scope('signal_to_magnitude_mel'):
            P, L = int(self.W - self.H) // 2, shape(signal)[1]
            signal = tf.pad(signal, ((0,0), (P, -L%self.H + P)), mode='REFLECT')

            stft = tf.signal.stft(signal, self.W, self.H, self.FFT)
            spectrogram = tf.abs(stft)
            magnitude = spectrogram/log_scaler
            magnitude = tf.where(magnitude > 1., 1.+tf.math.log(magnitude), magnitude)

            mel_weights_matrix = self.get_mag_to_mel_matrix()

            mel_spectrogram = tf.tensordot(spectrogram, mel_weights_matrix, [-1, 0])
            mel = mel_spectrogram/log_scaler
            mel = tf.where(mel > 1., 1.+tf.math.log(mel), mel)
        return magnitude, mel
    """

    def get_mag_to_mel_matrix(self):
        #return
        #a = tf.signal.linear_to_mel_weight_matrix(self.M, self.FFT//2+1, self.SR, self.Low_Clip, self.High_Clip, dtype=tf.float32)
        #mel_weights_matrix =
        mel_weights_matrix = librosa_mel_fn(sr=self.SR, n_fft=self.FFT, n_mels=self.M, fmin=self.Low_Clip, fmax=self.High_Clip)
        #cprint(a, color='red')
        #cprint(b.shape, color='blue')
        return mel_weights_matrix


    ### nsf-hifigan setting
    def sig_to_mag_mel(self, signal, minval=1e-5, log_constant=1):
        self.MinMel = np.log(minval)
        with tf.variable_scope('signal_to_melspec'):
            # total length should be win_length + (array_length - 1) * hop_length
            # L_sig = W + (L_m - 1) * H
            P, L = int(self.W - self.H) // 2, shape(signal)[1]
            signal = tf.pad(signal, ((0,0), (P, -L%self.H + P)), mode='REFLECT')

            stft = tf.signal.stft(signal, self.W, self.H, self.FFT) #/ (self.FFT // 2 + 1)
            spectrogram = tf.sqrt(tf.pow(tf.real(stft), 2) + tf.pow(tf.imag(stft), 2) + 1e-9)

            #with tf.control_dependencies([tf_print('tfspec', spectrogram, color='red')]): spectrogram=tf.identity(spectrogram)

            #spec = torch.stft(signal, self.FFT, hop_length=self.H, win_length=self.W, window=torch.hann_window(self.W),
            #        center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
            #spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

            #cprint(('tcspec', spec), color='blue')

            mel_weights_matrix = self.get_mag_to_mel_matrix()

            mel_spectrogram = tf.tensordot(spectrogram, mel_weights_matrix, [-1, -1]) # * self.M
            mel = tf.log(tf.where(mel_spectrogram < minval, minval * tf.ones_like(mel_spectrogram), mel_spectrogram) * log_constant)
            # log mel to log10 mel
            mel = 0.434294 * mel
            self.MinMel = 0.434294 * np.log(minval)

        return spectrogram, mel


    '''
    ### hifigan setting
    def sig_to_mag_mel(self, signal, minval=1e-5):
        self.MinMel = np.log(minval)
        with tf.variable_scope('signal_to_melspec'):
            # total length should be win_length + (array_length - 1) * hop_length
            # L_sig = W + (L_m - 1) * H
            P, L = int(self.W - self.H) // 2, shape(signal)[1]
            signal = tf.pad(signal, ((0,0), (P, -L%self.H + P)), mode='REFLECT')

            stft = tf.signal.stft(signal, self.W, self.H, self.FFT)
            spectrogram = tf.abs(stft)
            magnitude = tf.log(tf.where(spectrogram < minval, minval * tf.ones_like(spectrogram), spectrogram))

            mel_weights_matrix = self.get_mag_to_mel_matrix()

            mel_spectrogram = tf.einsum('...s,sm->...m', spectrogram, mel_weights_matrix)
            mel = tf.log(tf.where(mel_spectrogram < minval, minval * tf.ones_like(mel_spectrogram), mel_spectrogram))

        return magnitude, mel


    def get_mag_to_mel_matrix(self, f_min=0.0, f_sp=200.0/3, min_log_hz=1000.0, logstep=np.log(6.4) / 27.0):
        #mel_weights_matrix = tf.signal.linear_to_mel_weight_matrix(
        #        self.M, self.FFT//2+1, self.SR, self.Low_Clip, self.High_Clip, dtype=tf.float32)
        #mel_weights_matrix /= self.Log_Scale

        #mel_weights_matrix = librosa_mel_fn(self.SR, self.FFT, self.M, self.Low_Clip, self.High_Clip)
        # TODO librosa has numbas version issue, so implemented explitly
        weights = np.zeros((self.M, int(1 + self.FFT // 2)), dtype=np.float32)
        #fftfreqs = fft_frequencies(sr=self.SR, n_fft=self.FFT)
        # TODO from librosa.core.convert fft_frequencies
        fft_freqs = np.fft.rfftfreq(n=self.FFT, d=1.0/self.SR)
        # TODO from librosa.core.convert mel_frequencies
        min_log_mel = (min_log_hz - f_min) / f_sp
        min_mel = (self.Low_Clip - f_min) / f_sp
        if self.Low_Clip >= min_log_hz: min_mel = min_log_mel + np.log(self.Low_Clip / min_log_hz) / logstep
        max_mel = (self.High_Clip - f_min) / f_sp
        if self.High_Clip >= min_log_hz: max_mel = min_log_mel + np.log(self.High_Clip / min_log_hz) / logstep
        mels = np.linspace(min_mel, max_mel, self.M + 2)
        mel_freqs = f_min + f_sp * mels
        log_t = mels >= min_log_mel
        mel_freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        fdiff = np.diff(mel_freqs)
        ramps = np.subtract.outer(mel_freqs, fft_freqs)

        for i in range(self.M):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            weights[i] = np.maximum(0, np.minimum(lower, upper))

        enorm = 2.0 / (mel_freqs[2 : self.M + 2] - mel_freqs[:self.M])
        weights *= enorm[:, np.newaxis]

        if not np.all((mel_freqs[:-2] == 0) | (weights.max(axis=1) > 0)):
            warnings.warn(
                    "Empty filters detected in mel frequency basis. "
                    "Some channels will produce empty responses. "
                    "Try increasing your sampling rate (and fmax) or "
                    "reducing n_mels.",
                    stacklevel=2,
                    )

        mel_weights_matrix = tf.transpose(tf.convert_to_tensor(weights), [1, 0])

        return mel_weights_matrix
    '''

    def get_drop_mask(self, mel_inputs, drop_value=4.):
        mean_inputs = tf.reduce_mean(mel_inputs, axis=-1)
        drop_mask = tf.where(mean_inputs < drop_value, tf.zeros_like(mean_inputs), tf.ones_like(mean_inputs))
        return drop_mask

    def compression(self, mel_inputs, drop_value=4.):
        drop_mask = self.get_drop_mask(mel_inputs, drop_value=drop_value)
        sorting_order = tf.argsort(drop_mask, axis=-1, direction='DESCENDING', stable=True)
        mel_sorted = tf.gather_nd(mel_inputs, sorting_order[Ellipsis, None], batch_dims=1)
        original_order = tf.gather_nd(tf.range(shape(mel_inputs)[-2])[None], sorting_order[Ellipsis, None], batch_dims=1)
        value_mask = tf.sort(drop_mask, axis=-1, direction='DESCENDING')
        return mel_sorted, value_mask, original_order, sorting_order

    def decompression(self, mel_sorted, original_order):
        recovering_order = tf.argsort(orignial_order, axis=-1, direction='ASCENDING', stable=True)
        mel_recover = tf.gather_nd(mel_sorted, recovering_order[Ellipsis, None], batch_dims=1)
        return mel_recover

    def get_optimizer(self):
        with tf.name_scope('optimizer'):
            global_step = tf.to_float(self.global_step)
            LR = self.LR
            if exists(self.Dec_Alpha) and exists(self.Dec_Steps) and self.Dec_Alpha < 1:
                LR_decayed = tf.train.cosine_decay(self.LR, global_step, self.Dec_Steps, self.Dec_Alpha)
            tf.summary.scalar('learning_rate', LR_decayed)
            opt_kwargs = {
                    'learning_rate': LR_decayed,
                    'beta1': self.Adam_B1,
                    'beta2': self.Adam_B2,
                    'epsilon': self.Adam_Eps,
                    }
            if exists(self.W_Decay) and self.W_Decay > 0:
                optimizer = AdamWeightDecayOptimizer(weight_decay_rate=self.W_Decay, **opt_kwargs)
            else:
                optimizer = tf.train.AdamOptimizer(**opt_kwargs)
        return optimizer



class AdamWeightDecayOptimizer(tf.compat.v1.train.AdamOptimizer):
    """
    Adam optimizer with weight decay.
    Note that the variables must be optimized using this optimizer will
    have to include 'kernel' or 'weights' as a substring in variable name
    for the optimizer to apply weight decay regularization.
    """
    def __init__(self, weight_decay_rate, *args, **kwargs):
        super(AdamWeightDecayOptimizer, self).__init__(*args, **kwargs)
        self._weight_decay_rate = weight_decay_rate

    def _prepare(self):
        super(AdamWeightDecayOptimizer, self)._prepare()
        self._decay_slots = []

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        for var in self.variables():
            if 'kernel' or 'weights' in var.name:
                self._decay_slots.append(self.get_slot(var, 'weight_decay'))
        return super(AdamWeightDecayOptimizer, self).apply_gradients(grads_and_vars, global_step, name)

    def _resource_apply_dense(self, grad, var):
        adjusted_grad = grad + self._weight_decay_rate * self.get_slot(var, 'weight_decay')
        var_update = super(AdamWeightDecayOptimizer, self)._resource_apply_dense(adjusted_grad, var)
        return var_update

    def _resource_apply_sparse(self, grad, var, indices):
        adjusted_grad = grad + self._weight_decay_rate * self.get_slot(var, 'weight_decay')
        var_update = super(AdamWeightDecayOptimizer, self)._resource_apply_sparse(adjusted_grad, var, indices)
        return var_update


    #def sig_to_code(self, signal):
    #    # pretrained only for 24khz
    #    encoder = EncodecModel.encodec_model_24khz()
    #    model.set_target_bandwidth(6.0)
    #    signal = convert_audio(

