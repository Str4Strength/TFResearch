import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from ..Neural_Network import *
from .attention_block import *



# causal, l2-attention based discriminator for lipschitz condition
def discriminator(
        tensor,
        reference,
        condition,
        features,
        kernels,
        frequency_strides,
        frequency_position = True,
        groups = 1,
        activation = mish,
        mask = None,
        lrmul = 1.0
        weight_function = spectral_norm,
        train = True,
        scope = 'disc',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):



