import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from .function import *
from .variable_setting import *
from .linear import *



def conditional_weight(
        condition,
        kernel,
        in_features,
        out_features,
        groups = 1,
        init = None,
        lrmul = 1.0,
        quantization = 0.0,
        function = None,
        trainable = True,
        scope = 'cond_kerw',
        **kwargs
        ):
    with tf.variable_scope(scope):
        prod_kernel = int(np.prod(kernel))
        weight = weight_(
                shape = [prod_kernel, prod_kernel, in_features // groups, out_features // groups],
                dtype = condition.dtype,
                init = init,
                lrmul = lrmul,
                function = function,
                trainable = trainable,
                **kwargs
                )
        if trainable: weight = quantization_noise(
                weight = weight,
                in_features = in_features // groups,
                out_features = out_features,
                p = quantization,
                blocks = prod_kernel * prod_kernel
                )

        score, _ = linear(
                tensor = condition,
                in_features = shape(condition)[-1],
                out_features = prod_kernel,
                lrmul = lrmul,
                quantization = quantization,
                quantization_blocks = shape(condition)[-1],
                weight_function = function,
                trainable = trainable,
                scope = 'affine'
                )
        score = tf.reshape(tf.nn.softmax(score, axis = -1), [shape(condition)[0], prod_kernel])

        weight = tf.einsum('bk,kwio->bwio', score, weight)
        if not isinstance(kernel, int):
            weight =  tf.reshape(weight, (shape(condition)[0], *kernel, in_features, out_features))

        return weight


