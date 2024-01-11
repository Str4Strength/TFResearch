import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from ..Neural_Network import *




def shortcut(
        tensor,
        features,
        mask = None,
        quantization = 0.0,
        quantization_blocks = 8,
        weight_function = None,
        bias_function = None,
        trainable = True,
        scope = 'shortcut'
        ):
    with tf.variable_scope(scope):

        if features == shape(tensor)[-1]:
            sc_out, sc_mask = tensor, mask
        else:
            sc_out, sc_mask = linear(
                    tensor = tensor,
                    out_features = int(features),
                    mask = mask,
                    quantization = quantization,
                    quantization_blocks = quantization_blocks,
                    weight_function = weight_function,
                    bias_function = bias_function,
                    trainable = trainable,
                    scope = 'proj'
                    )

        return sc_out, sc_mask

