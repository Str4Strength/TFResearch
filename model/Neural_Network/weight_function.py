import math
import warnings

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .function import *


def spectral_normalize(weight, power_iteration_rounds=1, trainable=True, name=None):
    with tf.variable_scope(name, 'spectral_norm'):
        w_shape = shape(weight)
        w = tf.reshape(weight, [-1, w_shape[-1]])
        u_var = tf.get_variable('spectral_norm_u', shape=[shape(w)[0], 1],
                dtype=w.dtype, initializer=tf.random_normal_initializer,
                trainable=False)
        u = u_var

        for _ in range(power_iteration_rounds):
            v = tf.nn.l2_normalize(tf.einsum('ad,ao->do', w, u))
            u = tf.nn.l2_normalize(tf.einsum('ad,do->ao', w, v))

        if trainable:
            with tf.control_dependencies([u_var.assign(u, name='update_u')]):
                u = tf.identity(u)

        u = tf.stop_gradient(u)
        v = tf.stop_gradient(v)

        spectral_norm = tf.einsum('od,do->', tf.einsum('ao,ad->od', u, w), v)
        #spectral_norm.shape.assert_is_fully_defined()
        #spectral_norm.shape.assert_is_compatible_with([1,1])

        normalized = weight / tf.maximum(spectral_norm, 1e-8)
    return tf.reshape(normalized, w_shape)


