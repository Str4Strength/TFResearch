# import base modules
import os
import re
import sys
import glob
import json
import fire
import time
import random
import itertools
import importlib

# import main modules
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
import soundfile as sf
#import lpips_tf

# import functions only
from tqdm import tqdm
from operator import add
from pprint import pprint
from itertools import cycle
from termcolor import colored, cprint
from multiprocessing import Process, Queue, Manager

# import custom modules
#from ..train.meta import *
from ..audio_utils import sig_to_mag_mel

# import outer custom modules
from ..train.hyper_params import hyper_params as hp
from ..sources import gen_net as G

from ..variable_loader import *

from ..nsf_hifigan import NsfHifiGAN, get_pitch_crepe
from ..nsf_hifigan.hparams import hparams, set_hparams

GEN = hp.generator


def get_signal(file_path, dtype='float32', normalize=True, norm_dBFS=-23.0):
    sig, sr = sf.read(file_path)
    if sr != hp.sampling_rate: sig = scipy.signal.resample(sig, int(np.ceil(len(sig)/(sr/hp.sampling_rate))))
    #stereo = (sig.ndim > 1) and (np.shape(sig)[-1] == 2)
    #if stereo: sig = np.mean(sig, axis=-1)
    if normalize: sig *= (10**(norm_dBFS/20) / max(1e-9, np.sqrt(np.mean(np.square(sig)))))
    return sig


def test(case, mel_folder='./mels', dest_folder='./wavs'):
    set_hparams('/home/junwoo/CoreResearch/model/nsf_hifigan/config_nsf.yaml')
    vocoder = NsfHifiGAN(device='cpu')#devices)

    spks = list(map(lambda f: f.split('/')[-1], glob.glob(os.path.join(hp.test_path, '*'))))
    speakers, depth = [], 0
    while speakers == []:
        speakers = [spk for spk in spks if len(glob.glob(os.path.join(hp.test_path, spk, ('*/'*depth) + '*.wav')))
                >= hp.batch_size]
        depth += 1

    wav = tf.placeholder(tf.float32, (None, None), 'sig_cnt')
    #w_s = tf.placeholder(tf.float32, (None, None), 'sig_sty')
    #lat = tf.placeholder(tf.float32, (None, MAP.dims[-1]), 'lat')
    mel_c = tf.placeholder(tf.float32, (None, None, hp.melfreq_size), 'mel_cnt')
    mel_s = tf.placeholder(tf.float32, (None, None, hp.melfreq_size), 'mel_sty')
    pit = tf.placeholder(tf.float32, (None, None, 1), 'pitch')

    g_kwargs = hp.generator.copy()
    del g_kwargs['load_case']

    gnet = G.Network(min_values=0.434294 * tf.log(1e-5), **g_kwargs)

    draw = gnet(mel_c, mel_s, pit, train=False, reuse=tf.AUTO_REUSE)
    cprint(draw, color='blue')

    to_mel = sig_to_mag_mel(
            wav, hp.sampling_rate, hp.window_size, hp.hop_size, hp.fft_size, hp.melfreq_size, hp.low_clip_hertz, hp.high_clip_hertz)[1]

    loads = get_loader([GEN.scope])
    log_path = '/home/junwoo/CoreResearch/logs'


    hpdict = {
            'audio_sample_rate': hp.sampling_rate,
            'f0_min': 40.0,
            'f0_max': 1100.0,
            'f0_bin': 256,
            'hop_size': hp.hop_size,
            }



    with tf.Session() as sess:
        load_variables_to_session(sess, loads, os.path.join(log_path, f'{case}'))

        cnts = glob.glob('/danube/datasets/naevis_441h/secured_44100/*.wav')
        already_generated = [fname.split('/')[-1].split('content_')[-1]
                for fname in glob.glob('/home/junwoo/CoreResearch/wavs/content_*.wav')]
        cprint(f'before removing generated files in queue : {len(cnts)}', color='green')
        cnts = [f for f in cnts if f not in already_generated]
        cprint(f'after removing generated files in queue : {len(cnts)}', color='green')
        with open('/home/junwoo/naevis.json', 'r') as f: nvs = json.load(f)
        stys = [d["file"] for d in nvs[:3]]

        pairs = []
        for sfile in stys:
            for cfile in cnts:
                pairs.append((cfile, sfile))

        for fc, fs in pairs:

            sig_c, sig_s = get_signal(fc)[None], get_signal(fs)[None]
            cprint(sig_c.shape, color='red')

            c_stereo, s_stereo = len(sig_c.shape) > 2, len(sig_s.shape) > 2
            if s_stereo: sig_s = np.mean(sig_s, axis=-1)

            c_sigs = [sig_c[Ellipsis, 0], sig_c[Ellipsis, 1]] if c_stereo else [sig_c]
            cprint(c_sigs, color='yellow')

            mels_g, sigs_g, mels_c, sigs_c = [], [], [], []
            for sig_c in c_sigs:

                mel_cnt = sess.run(to_mel, feed_dict={wav: sig_c})
                mels_c.append(mel_cnt)
                cprint(mel_cnt.shape, color='red')

                mel_sty = sess.run(to_mel, feed_dict={wav: sig_s})
                cprint(mel_sty.shape, color='magenta')

                #lat_sty = sess.run(get_style, feed_dict={w_s: sig_s})

                """
                sig_s = np.concatenate([get_signal(file_s) for file_s in fs], axis=0)[None]
                """

                cprint((fc, fs), 'green')

                stride = hp.meltime_size // 2
                cnt_length = mel_cnt.shape[1]
                min_len_pad = max(hp.meltime_size - cnt_length, 0)

                mel_cnt = np.pad(mel_cnt, ((0, 0), (0, -cnt_length%stride + min_len_pad), (0, 0)))
                f0_c, _ = get_pitch_crepe(sig_c[0], mel_cnt[0, :cnt_length], hparams=hpdict)
                f0_c = np.pad(f0_c, (0, -cnt_length%stride + min_len_pad))

                ##### process splitted into trained size #####
                mlen_sc = mel_cnt.shape[1]

                mel_gen = []
                for a in range(mlen_sc // stride - 1):
                    start_id = stride * a
                    end_id = start_id + hp.meltime_size
                    cprint((start_id, end_id), color='magenta')

                    mc = mel_cnt[:, stride * a : min(end_id, mlen_sc)]
                    pc = f0_c[stride * a : min(end_id, mlen_sc)][None, :, None]
                    #cprint(pc, color='cyan')
                    cprint(len(mc[0]), color='blue')
                    cprint((mc, mel_sty, pc), color='cyan')
                    cprint((mel_c, mel_s, pit), color='magenta')

                    mg, _ = sess.run(
                            #gnet(mel_c, mel_s, pit, train=False, reuse=True),
                            draw,
                            feed_dict={mel_c: mc, mel_s: mel_sty, pit: pc})
                    if a != mlen_sc // stride - 2:
                        mg = mg[:, :-(stride//2)]
                    if a != 0:
                        mg = mg[:, (stride//2) :]
                    cprint(len(mg[0]), color='cyan')
                    mel_gen.append(mg)


                mel_gen = np.concatenate(mel_gen, axis=1)[:, :cnt_length]
                mels_g.append(mel_gen)
                cprint(mel_gen.shape, color='green')
                """

                mel_gen = sess.run(solver(mel, lat), feed_dict={mel: mel_cnt, lat: lat_sty})[0]
                mels_g.append(mel_gen)

                """

                #f0_c, _ = get_pitch_crepe(sig_c[0], mel_cnt[0, :cnt_length], hparams=hpdict)
                f0_c = f0_c[:cnt_length]
                f0_s, _ = get_pitch_crepe(sig_s[0], mel_sty[0], hparams=hpdict)

                wav_c = vocoder.spec2wav_torch(torch.Tensor(mel_cnt[:, :cnt_length]), f0 = torch.Tensor(f0_c)[None])
                wavs_c.append(wav_c)

                wav_s = vocoder.spec2wav_torch(torch.Tensor(mel_sty), f0 = torch.Tensor(f0_s)[None])

                wav_g = vocoder.spec2wav_torch(torch.Tensor(mel_gen), f0 = torch.Tensor(f0_c)[None])
                wavs_g.append(wav_g)

            wav_c = np.stack(wavs_c, axis=-1) if c_stereo else wavs_c[0]
            wav_g = np.stack(wavs_g, axis=-1) if c_stereo else wavs_g[0]
            mel_c = np.mean(mels_c, axis=0) if c_stereo else mels_c[0]
            mel_g = np.mean(mels_g, axis=0) if c_stereo else mels_g[0]

            # TODO mel numpy로 저장
            mel_dir = './mels'
            os.makedirs(mel_dir, exist_ok=True)

            cname = fc.split('/')[-1][:-4]
            sname = fs.split('/')[-1][:-4]
            #sname = 'tongil'

            # content 원본
            save_path = os.path.join(mel_dir, f'content_{cname}.npy')
            np.save(save_path, np.transpose(mel_cnt[0]))

            # negative style 원본
            save_path = os.path.join(mel_dir, f'style_{sname}.npy')
            np.save(save_path, np.transpose(mel_sty[0]))

            # 변환본
            save_path = os.path.join(mel_dir, f'convert_{cname}_sty_{sname}.npy')
            np.save(save_path, np.transpose(mel_gen[0]))


            sf.write(os.path.join(dest_folder, f'content_{cname}.wav'), wav_c, 44100, 'PCM_32')
            sf.write(os.path.join(dest_folder, f'style_{sname}.wav'), wav_s, 44100, 'PCM_32')
            sf.write(os.path.join(dest_folder, f'convert_{cname}_sty_{sname}.wav'), wav_g, 44100, 'PCM_32')

