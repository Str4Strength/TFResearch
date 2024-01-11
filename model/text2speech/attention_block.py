import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from ..Neural_Network import *
from .block_setting import *



# core tensor : repeated text-oriented tokens, matched into speech
# condition tensor : speaker's style, for speaking tendency
# cross condition tensor : original text, for context understanding



def conditional_gamma_beta(
        tensor,
        groups,
        quantization = 0.0,
        weight_function = None,
        trainable = True,
        scope = 'cond'
        ):
    with tf.variable_scope(scope):
        gamma_beta, _ = linear(
                tensor = tf.reshape(tensor, [shape(tensor)[0], shape(tensor)[-1]]),
                out_features = 2 * groups,
                quantization = quantization,
                quantization_blocks = shape(tensor)[-1],
                weight_function = weight_function,
                trainable = trainable,
                scope = 'gmbt'
                )
        gamma, beta = gamma_beta[:, None, :groups, None], gamma_beta[:, None, groups:, None]

        return gamma, beta



def attention_block(
        tensor,
        companion = None,
        condition = None,
        iteration = 5,
        features = 512,
        core_operation = general_attention,
        head = 8,
        window = None,
        kernel = 5,
        weight_groups = 1,
        normalization_groups = 16,
        causality = False,
        conditional_convolution = False,
        activation = gelu,
        mask = None,
        mask_companion = None,
        saved_state = None,
        normalize = True,
        residualize = True,
        scale_shortcut = False,
        quantization = 0.0,
        weight_function = None,
        trainable = True,
        scope = 'attn_blk',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse):

        feature_maps, attention_maps = [], []
        # abs pos enc skipped since rotary pos enc internally applied in attention
        for n in range(iteration):

            with tf.variable_scope(f'{n:02d}'):

                with tf.variable_scope('major_block'):

                    use_attn_bias = core_operation == 'general_attention' and isinstance(window, int)
                    if use_attn_bias:
                        attention_bias = tf.ones((shape(tensor)[1], shape(tensor)[1]))
                        attention_bias = tf.linalg.band_part(attention_bias, (window - 1) // 2, window // 2)
                        attention_bias = 1.0 - attention_bias
                        attention_bias = -1e9 * attention_bias

                    residual, _, attention_map = attention(
                            query = tensor,
                            key = tensor,
                            value = tensor,
                            out_features = features,
                            core_operation = core_operation,
                            head = head,
                            window = window,
                            mask_query = mask,
                            mask_key = mask,
                            mask_value = mask,
                            causality = causality,
                            saved_state = getattr(saved_state, 'attn', None),
                            attention_bias = None if not use_attn_bias else attention_bias,
                            quantization = quantization,
                            quantization_blocks = features,
                            query_weight_function = weight_function,
                            key_weight_function = weight_function,
                            value_weight_function = weight_function,
                            trainable = trainable,
                            scope = 'attn'
                            )
                    attention_maps.append(attention_map)

                    if normalize:
                        if exists(condition):
                            gamma, beta = conditional_gamma_beta(
                                    tensor = condition,
                                    groups = normalization_groups,
                                    quantization = quantization,
                                    weight_function = weight_function,
                                    trainable = trainable,
                                    scope = 'norm'
                                    )
                        else:
                            gamma, beta = None, None

                        residual = normalization(
                                tensor = residual,
                                groups = normalization_groups,
                                gamma = gamma,
                                beta = beta,
                                mask = mask,
                                gamma_function = weight_function,
                                trainable = trainable,
                                scope = 'norm'
                                )

                    if residualize:
                        skip, mask = shortcut(
                                tensor = tensor,
                                features = features,
                                mask = mask,
                                quantization = quantization,
                                quantization_blocks = features,
                                weight_function = weight_function,
                                trainable = trainable,
                                scope = 'scut'
                                )
                        if scale_shortcut:
                            scale = weight_(
                                    shape = [],
                                    dtype = tensor.dtype,
                                    init = tf.zeros_initializer,
                                    trainable = trainable,
                                    scope = 'resc'
                                    )
                            scale = tf.minimum(tf.math.abs(scale), 1)
                        else:
                            scale = 1
                        tensor = scale * skip + residual
                    else:
                        tensor = residual

                if exists(companion):

                    with tf.variable_scope('vital_block'):

                        residual, _, attention_map = attention(
                                query = tensor,
                                key = companion,
                                value = companion,
                                out_features = features,
                                core_operation = core_operation,
                                head = head,
                                window = window,
                                mask_query = mask,
                                mask_key = mask_companion,
                                mask_value = mask_companion,
                                saved_state = getattr(saved_state, 'attn', None),
                                quantization = quantization,
                                quantization_blocks = features,
                                query_weight_function = weight_function,
                                key_weight_function = weight_function,
                                value_weight_function = weight_function,
                                trainable = trainable,
                                scope = 'attn'
                                )
                        attention_maps.append(attention_map)

                        if normalize:
                            if exists(condition):
                                gamma, beta = conditional_gamma_beta(
                                        tensor = condition,
                                        groups = normalization_groups,
                                        quantization = quantization,
                                        weight_function = weight_function,
                                        trainable = trainable,
                                        scope = 'norm'
                                        )
                            else:
                                gamma, beta = None, None

                            residual = normalization(
                                    tensor = residual,
                                    groups = normalization_groups,
                                    gamma = gamma,
                                    beta = beta,
                                    mask = mask,
                                    gamma_function = weight_function,
                                    trainable = trainable,
                                    scope = 'norm'
                                    )

                        if residualize:
                            if scale_shortcut:
                                scale = weight_(
                                        shape = [],
                                        dtype = tensor.dtype,
                                        init = tf.zeros_initializer,
                                        trainable = trainable,
                                        scope = 'resc'
                                        )
                                scale = tf.minimum(tf.math.abs(scale), 1.)
                            else:
                                scale = 1
                            tensor = scale * tensor + residual
                        else:
                            tensor = residual

                with tf.variable_scope('minor_block'):

                    if exists(condition) and conditional_convolution:
                        r = tf.transpose(tensor, (1, 0, 2))
                        r = tf.reshape(r, (1, shape(r)[0], -1))
                        if exists(mask):
                            r_m = tf.transpose(mask, (1, 0, 2))
                            r_m = tf.reshape(mask, shape(r))

                        xpnd_weight = conditional_weight(
                                condition = condition,
                                kernel = kernel,
                                in_features = features,
                                out_features = 2 * features,
                                groups = weight_groups,
                                quantization = quantization,
                                function = weight_function,
                                trainable = trainable,
                                scope = 'xpnd_weight'
                                ) # b, k, i, o
                        xpnd_weight = tf.transpose(xpnd_weight, (1, 2, 0, 3))
                        xpnd_weight = tf.reshape(xpnd_weight, (kernel, features // weight_groups, -1))     # k, i, b * o

                        xpnd_bias = bias_(
                                shape = (1, 2 * features),
                                dtype = tensor.dtype,
                                trainable = trainable,
                                scope = 'xpnd_bias'
                                )
                        xpnd_bias = tf.tile(xpnd_bias, (shape(condition)[0], 1))
                        xpnd_bias = tf.reshape(xpnd_bias, (1, 1, -1))

                    else:
                        r, r_m, xpnd_weight, xpnd_bias = tensor, mask, None, None


                    if causality:
                        xpnd, xpnd_m = getattr(saved_state, 'xpnd', None), getattr(saved_state, 'xpnd_mask', None)
                        if exists(xpnd):
                            r = link_memory(r, None, tensor_saved = xpnd)
                            setattr(saved_state, 'xpnd', r[:, - kernel + 1:])
                        else:
                            r = tf.pad(r, ((0, 0), (kernel - 1, 0), (0, 0)))

                    residual, _ = convolution(
                            tensor = r,
                            rank = 1,
                            filters = 2 * features,
                            kernel = kernel,
                            padding = 'VALID' if causality else 'SAME',
                            groups = weight_groups,
                            weight = xpnd_weight,
                            bias = xpnd_bias,
                            quantization = quantization,
                            quantization_blocks = np.prod(kernel),
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'xpnd'
                            )
                    res_mask = reconstruct_mask(shape(residual)[-1], r_m)
                    if exists(mask): residual *= res_mask

                    residual = activation(residual)
                    feature_maps.append(residual)

                    if exists(condition) and conditional_convolution:
                        sqze_weight = conditional_weight(
                                condition = condition,
                                kernel = kernel,
                                in_features = 2 * features,
                                out_features = features,
                                groups = weight_groups,
                                quantization = quantization,
                                function = weight_function,
                                trainable = trainable,
                                scope = 'sqze_weight'
                                ) # b, k, i, o
                        sqze_weight = tf.transpose(sqze_weight, (1, 2, 0, 3))
                        sqze_weight = tf.reshape(sqze_weight, (kernel, 2 * features // weight_groups, -1))

                        sqze_bias = bias_(
                                shape = (1, features),
                                dtype = tensor.dtype,
                                trainable = trainable,
                                scope = 'sqze_bias'
                                )
                        sqze_bias = tf.tile(sqze_bias, (shape(condition)[0], 1))
                        sqze_bias = tf.reshape(sqze_bias, (1, 1, -1))

                    else:
                        sqze_weight, sqze_bias = None, None


                    if causality:
                        sqze, sqze_m = getattr(saved_state, 'sqze', None), getattr(saved_state, 'sqze_mask', None)
                        if exists(sqze):
                            r = link_memory(residual, None, tensor_saved = sqze)
                            setattr(saved_state, 'sqze', r[:, - kernel + 1:])
                        else:
                            r = tf.pad(residual, ((0, 0), (kernel - 1, 0), (0, 0)))
                    else:
                        r = residual

                    residual, _ = convolution(
                            tensor = r,
                            rank = 1,
                            filters = features,
                            kernel = kernel,
                            padding = 'VALID' if causality else 'SAME',
                            groups = weight_groups,
                            weight = sqze_weight,
                            bias = sqze_bias,
                            quantization = quantization,
                            quantization_blocks = np.prod(kernel),
                            weight_function = weight_function,
                            trainable = trainable,
                            scope = 'sqze'
                            )

                    if exists(condition) and conditional_convolution:
                        residual = tf.reshape(residual, (shape(residual)[1], shape(residual)[-1] // features, features))
                        residual = tf.transpose(residual, (1, 0, 2))

                    if exists(mask): residual *= mask

                    if normalize:
                        if exists(condition):
                            gamma, beta = conditional_gamma_beta(
                                    tensor = condition,
                                    groups = normalization_groups,
                                    quantization = quantization,
                                    weight_function = weight_function,
                                    trainable = trainable,
                                    scope = 'norm'
                                    )
                        else:
                            gamma, beta = None, None

                        residual = normalization(
                                tensor = residual,
                                groups = normalization_groups,
                                gamma = gamma,
                                beta = beta,
                                mask = mask,
                                gamma_function = weight_function,
                                trainable = trainable,
                                scope = 'norm'
                                )

                    if residualize:
                        if scale_shortcut:
                            scale = weight_(
                                    shape = [],
                                    dtype = tensor.dtype,
                                    init = tf.zeros_initializer,
                                    trainable = trainable,
                                    scope = 'resc'
                                    )
                            scale = tf.minimum(tf.math.abs(scale), 1)
                        else:
                            scale = 1
                        tensor = scale * tensor + residual
                    else:
                        tensor = residual

        return tensor, mask, feature_maps, attention_maps



