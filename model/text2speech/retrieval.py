import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *



def retrieval(
        query,
        capacity,
        size,
        dtype = tf.float32,
        zero_pad = True,
        scale = True,
        trainable = True,
        scope = 'tknz',
        **retrieval_kwargs
        ):
    with tf.variable_scope(scope):
        retrieval_index = weight_(
                shape = [capacity - 1 if zero_pad else capacity, size],
                dtype = dtype,
                trainable = trainable,
                scope = 'lkup',
                **retrieval_kwargs
                )
        if zero_pad: retrieval_index = tf.concat(
                values = [tf.zeros(shape = [1, size], dtype = dtype), retrieval_index],
                axis = 0
                )

        tensor = tf.gather(params = retrieval_index, indices = query, batch_dims = 0)

        if scale: tensor *= tf.rsqrt(tf.cast(size, dtype))

        return tensor

