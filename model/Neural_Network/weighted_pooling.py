import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from .convolution import *
from .variable_setting import *



def weighted_pooling(
        tensor,
        rank,
        kernels,
        strides,
        padding = 'SAME',
        mask = None,
        weight = None,
        lrmul = 1.0,
        trainable = True,
        scope = 'pooling'
        ):
    with tf.variable_scope(scope):
        pool_mask = mask if exits(mask) else tf.ones_like(tensor)

        if not exists(weight):
            weight = weight_(
                    shape = kernels + [1, 1],
                    dtype = tensor.dtype,
                    lrmul = lrmul,
                    function = weight_function,
                    trainable = trainable
                    )

        shared_kwargs = {
                "rank": rank,
                "filters" : shape(tensor)[-1],
                "kernels" : kernels,
                "strides" : strides,
                "padding" : padding,
                "groups" : shape(tensor)[-1],
                }

        kernelized_sums, mask = convolution(
                tensor = tensor,
                mask = mask,
                weight = weight,
                bias = False,
                trainable = False,
                scope = 'pool',
                **shared_kwargs
                )

        kernelized_divs, _ = convolution(
                tensor = pool_mask,
                weight = weight,
                bias = False,
                trainable = False,
                scope = 'mask',
                **shared_kwargs
                )

        poolout = tf.where(tf.equal(kernelized_divs, 0), tf.zeros_like(kernelized_sums), kernelized_sums / kernelized_divs)
        return poolout, mask

