import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from librosa.filters import mel as librosa_mel_fn


# TODO nsf-hifigan fitting functions
def mag_mel_converter(
        sampling_rate,
        fft_size,
        melfreq_size,
        minfreq,
        maxfreq,
        ):
    matrix = librosa_mel_fn(sr=sampling_rate, n_fft=fft_size, n_mels=melfreq_size, fmin=minfreq, fmax=maxfreq)
    return matrix

def sig_to_mag_mel(
        signal,
        sampling_rate,
        window_size,
        hop_size,
        fft_size,
        melfreq_size,
        minfreq,
        maxfreq,
        minval=1e-5,
        log_constant=1,
        ):
    pad, len_ = int(window_size - hop_size) // 2, tf.shape(signal)[1]
    signal = tf.pad(signal, ((0, 0), (pad, -len_%hop_size + pad)), mode='REFLECT')

    stft = tf.signal.stft(signal, window_size, hop_size, fft_size)
    spectrogram = tf.sqrt(tf.pow(tf.real(stft), 2) + tf.pow(tf.imag(stft), 2) + 1e-9)

    mel_weights_matrix = mag_mel_converter(
            sampling_rate=sampling_rate, fft_size=fft_size, melfreq_size=melfreq_size, minfreq=minfreq, maxfreq=maxfreq)

    mel_spectrogram = tf.tensordot(spectrogram, mel_weights_matrix, [-1, -1]) # * self.M

    mel = tf.log(tf.where(mel_spectrogram < minval, minval * tf.ones_like(mel_spectrogram), mel_spectrogram) * log_constant)
    # log mel to log10 mel
    mel = 0.434294 * mel
    #self.MinMel = 0.434294 * np.log(minval)

    return spectrogram, mel


def normalize_audio(wave, dBFS):
    return wave * (10 ** (dBFS/20) / max(1e-8, np.sqrt(np.mean(np.square(wave)))))

