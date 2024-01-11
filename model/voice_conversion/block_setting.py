import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .Neural_Networks import *




def res_scale(value, dtype, trainable=True, scope='res_scale'):
    scale = tf.get_variable(scope, [], dtype=dtype, initializer=tf.constant_initializer(value=value),
            trainable=trainable)
    scale = tf.minimum(tf.math.abs(scale), 1)
    return scale, tf.sqrt(1 - (scale ** 2))



def shortcut(
        tensor,
        features,
        weight_function = None,
        bias_function = None,
        mask = None,
        blocks = 8,
        trainable = True,
        scope = 'shortcut'
        ):
    with tf.variable_scope('skip'):

        if features == shape(tensor)[-1]:
            sc_out, sc_mask = tensor, mask
        else:
            sc_out, sc_mask = linear(tensor, out_features = int(features), mask = mask, quantization = R_Q,
                    quantization_blocks = blocks, weight_function = weight_function, bias_function = bias_function,
                    trainable = trainable, scope = 'proj')

        return sc_out, sc_mask

