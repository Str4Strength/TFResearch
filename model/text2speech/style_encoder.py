import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *



def style_encoder(
        tensor,
        features = None,
        iteration = 1,
        mask = None,
        quantization = 0.0,
        trainable = True,
        scope = 'style_encoder',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse = reuse):
        B = shape(tensor)[0]
        tensor = tensor[Ellipsis, None]
        if exists(mask): mask = mask[Ellipsis, None]

        for n in range(iteration):
            with tf.variable_scope(f'{n:02d}'):
                tensor, _ = deconvolution(
                        tensor = tensor,
                        rank = 2,
                        filters = 1,
                        kernel = [3, 4],
                        stride = [1, 2],
                        quantization = quantization,
                        quantization_blocks = 12,
                        trainable = trainable,
                        scope = f'dc2d'
                        )
                tensor = mish(tensor)
                if exists(mask):
                    mask = tf.reshape(tf.tile(mask[Ellipsis, None, :], [1, 1, 1, 2, 1]), shape(tensor))
                    tensor *= mask

                tensor, _ = convolution(
                        tensor = tensor,
                        rank = 2,
                        filters = 1,
                        kernel = [4, 3],
                        stride = [2, 1],
                        quantization = quantization,
                        quantization_blocks = 12,
                        trainable = trainable,
                        scope = f'cv2d'
                        )
                tensor = mish(tensor)
                if exists(mask):
                    mask = mask[:, ::2]
                    tensor *= mask

        seed_token = weight_(
                shape = [1, 1, shape(tensor)[-2]],
                dtype = tensor.dtype,
                trainable = trainable,
                scope = 'seed'
                )

        tensor = tf.squeeze(tensor, axis = -1)
        if exists(mask): mask = tf.squeeze(mask, axis = -1)

        tensor, _, _ = attention(
                query = tf.tile(seed_token, (shape(tensor)[0], 1, 1)),
                key = tensor,
                value = tensor,
                out_features = features,
                head = shape(tensor)[-1] // 16,
                mask_key = mask,
                mask_value = mask,
                quantization = quantization,
                quantization_blocks = 8,
                trainable = trainable,
                scope = 'attn'
                )

        return tensor[:, 0]

