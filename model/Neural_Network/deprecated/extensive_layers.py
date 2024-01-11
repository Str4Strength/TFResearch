import re
import math
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
# from tensorflow.python.ops import variables as tf_variables

# from tensorflow.contrib.nccl.ops import gen_nccl_ops
# from tensorflow.contrib.framework import add_model_variable

from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.utils.argtools import log_once

from functools import partial
from termcolor import cprint

from .operations import *
from ..utils import tf_print

'''
def normalization(in_,
                  where,
                  mask=None,
                  n_batch_dims=1,
                  groups=1,
                  epsilon=1e-8,
                  momentum=0.99, # for BN
                  activation=None,
                  scale=True,
                  gamma_initializer=tf.ones_initializer(),
                  gamma_regularizer=None,
                  gamma_constraint=None,
                  gamma_normalizer=None,
                  shift=True,
                  beta_initializer=tf.zeros_initializer(),
                  beta_regularizer=None,
                  beta_constraint=None,
		  pre_condition=None,
                  pre_groups=1,
                  pre_use_bias=True,
                  pre_activation=None,
                  pre_weight_initializer=tf.glorot_uniform_initializer(),
                  pre_weight_regularizer=None,
                  pre_weight_constraint=None,
                  pre_weight_normalizer=None,
                  pre_bias_initializer=tf.zeros_initializer(),
                  pre_bias_regularizer=None,
                  pre_bias_constraint=None,
		  post_condition=None,
                  post_condition_mask=None,
                  post_gamma_groups=1,
                  post_gamma_use_bias=True,
                  post_gamma_activation=None,
                  post_gamma_weight_initializer=tf.glorot_uniform_initializer(),
                  post_gamma_weight_regularizer=None,
                  post_gamma_weight_constraint=None,
                  post_gamma_weight_normalizer=None,
                  post_gamma_bias_initializer=tf.zeros_initializer(),
                  post_gamma_bias_regularizer=None,
                  post_gamma_bias_constraint=None,
                  post_beta_groups=1,
                  post_beta_use_bias=True,
                  post_beta_activation=None,
                  post_beta_weight_initializer=tf.glorot_uniform_initializer(),
                  post_beta_weight_regularizer=None,
                  post_beta_weight_constraint=None,
                  post_beta_weight_normalizer=None,
                  post_beta_bias_initializer=tf.zeros_initializer(),
                  post_beta_bias_regularizer=None,
                  post_beta_bias_constraint=None,
		  trainable=True,
                  scope='normalization'):
    """
    where:
        batch -> batch normalization,
        convolution_batch -> batch normalization after convolution layer
        instance -> instance normalization
        layer -> layer normalization
        convolution_layer -> layer normalization after convolution layer
        group -> group normalization
        *** `adaptive` -> require post_condition, do not generate gamma and beta, only generate post-gamma and post-beta
        *** no `adaptive` but post_condition given -> conditional normalization
    """
    where = where.upper()
    if 'ADAPTIVE' in where and post_condition is None: raise ValueError("'ADAPTIVE' normalization requires post_condition vertor.")
    is_batch = ('BATCH' in where)
    is_convolution = ('CONVOLUTION' in where)
    is_instance = ('INSTANCE' in where)
    is_group = ('GROUP' in where)
    is_layer = ('LAYER' in where)

    with tf.variable_scope(scope):
        #graph_name = tf.get_default_graph().get_name_scope()

        # shape information
        dtype = in_.dtype

        in_shape = shape_of(in_)
        B_axis = [axis for axis in range(n_batch_dims)]
        Bs = in_shape[:n_batch_dims]
        B = tf.reduce_prod(Bs)
        L = tf.reduce_prod(in_shape[n_batch_dims:-1])
        D_in = in_shape[-1]
        expanding = len(in_shape) - n_batch_dims > 2

        if mask == None: mask = tf.ones(in_shape[:-1], dtype=dtype)
        base_mask = mask

        if expanding or (n_batch_dims > 1):
            in_ = tf.reshape(in_, [B, L, D_in])
            mask = tf.reshape(mask, [B, L])

        if pre_condition is not None:
            pre_shape = shape_of(pre_condition)
            L_pre = tf.reduce_prod(pre_shape[n_batch_dims:-1])
            pre_condition = tf.reshape(pre_condition, [B, L_pre, pre_shape[-1]])
            pre_condition = tf.cond(tf.equal(L_pre, 1), lambda: tf.tile(pre_condition, [1, L, 1]), lambda: pre_condition)

            in_ = dense(tf.concat([in_, pre_condition], axis=-1) * mask[Ellipsis, None], in_shape[-1], groups=pre_groups, use_bias=pre_use_bias, activation=pre_activation,
                weight_initializer=pre_weight_initializer, weight_regularizer=pre_weight_regularizer, weight_constraint=pre_weight_constraint, weight_normalizer=pre_weight_normalizer,
                bias_initializer=pre_bias_initializer, bias_regularizer=pre_bias_regularizer, bias_constraint=pre_bias_constraint, scope='pre_dense', trainable=trainable)

        if is_instance or is_convolution:
            broad_axis = [1]
            broad_goal = ''
        else:
            broad_axis = []
            broad_goal = 'l'

        if is_batch:
            axis = [0] + broad_axis
            goal = broad_goal
        else:
            axis = broad_axis
            goal = 'b' + broad_goal

        if is_group:
            goal = goal + 'g'
        elif not is_layer:
            goal = goal + 'd'

        instance_bool = is_instance or tf.is_tensor(L)
        if is_group:
            G = min(groups, D_in)
            D_G = D_in//G
            scale_shift_shape = [1, L if (is_convolution and not instance_bool) else 1, G, 1]
            reshaper = [1 if is_batch else B, 1 if (is_convolution or is_instance) else L, G, 1]
            einsum_mask = tf.tile(mask[Ellipsis, None, None], [1, 1, G, D_G])
            start = "blgd"
            in_ = tf.reshape(in_, [B, L, G, D_G])
        else:
            scale_shift_shape = [1, L if (is_convolution and not instance_bool) else 1, 1 if is_layer else D_in]
            reshaper = [1 if is_batch else B, 1 if (is_convolution or is_instance) else L, 1 if is_layer else D_in]
            einsum_mask = tf.tile(mask[Ellipsis, None], [1, 1, D_in])
            start = "bld"

        if is_batch:
            # recorded mean, variance
            moving_mean = tf.get_variable('moving_mean', shape=scale_shift_shape, dtype=dtype, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('moving_variance', shape=scale_shift_shape, dtype=dtype, initializer=tf.ones_initializer, trainable=False)

            # mean, variance
            if trainable:
                reciprocal_mask_sum = tf.math.maximum(tf.einsum(start + "->" + goal, einsum_mask), epsilon) ** (-1.0)
                current_mean = tf.einsum(",".join([start, start, goal]) + "->" + goal, in_, einsum_mask, reciprocal_mask_sum)
                current_variance = tf.einsum(",".join([start, start, goal]) + "->" + goal, tf.math.square(in_ - tf.reshape(current_mean, reshaper)), einsum_mask, reciprocal_mask_sum)

                assign_shaper = reshaper[:1] + [1] + reshaper[2:]
                mean = ((1.0 - momentum) * tf.reshape(current_mean, reshaper)) + (momentum * tf.cast(moving_mean, dtype))
                assign_mean = ((1.0 - momentum) * tf.reshape(tf.einsum(",".join([goal, goal]) + "->" + re.sub("l", "", goal), current_mean, reciprocal_mask_sum),
                    assign_shaper)) + (momentum * tf.cast(moving_mean, dtype)) if tf.is_tensor(L) and not is_instance else mean
                divisor = tf.to_float(B) / tf.math.maximum(tf.to_float(B-1), epsilon)
                variance = ((1.0 - momentum) * divisor * tf.reshape(current_variance, reshaper)) + (momentum * tf.cast(moving_variance, dtype))
                assign_variance = ((1.0 - momentum) * divisor * tf.reshape(tf.einsum(",".join([goal, goal]) + "->" + re.sub("l", "", goal), current_variance, reciprocal_mask_sum),
                    assign_shaper)) + (momentum * tf.cast(moving_variance, dtype)) if tf.is_tensor(L) and not is_instance else variance

                with tf.control_dependencies([tf.assign(moving_mean, assign_mean), tf.assign(moving_variance, assign_variance)]):
                    out_ = (in_ - mean) * tf.math.rsqrt(tf.math.maximum(variance, epsilon))

            else:
                out_ = (in_ - tf.cast(moving_mean, dtype)) * tf.math.rsqrt(tf.math.maximum(tf.cast(moving_variance, dtype), epsilon))

        else:
            reciprocal_mask_sum = tf.math.maximum(tf.einsum(start + "->" + goal, einsum_mask), epsilon) ** (-1.0)
            mean = tf.reshape(tf.einsum(",".join([start, start, goal]) + "->" + goal, in_, einsum_mask, reciprocal_mask_sum), reshaper)
            delta = in_ - mean
            variance = tf.reshape(tf.einsum(",".join([start, start, goal]) + "->" + goal, tf.math.square(delta), einsum_mask, reciprocal_mask_sum), reshaper)

            out_ = delta * tf.math.rsqrt(tf.math.maximum(variance, epsilon))

        if post_condition is not None:
            post_shape = shape_of(post_condition)
            L_post = tf.reduce_prod(post_shape[n_batch_dims:-1])
            D_post = post_shape[-1]
            condition_shaper = [B, L_post, D_post]
            condition_reshaper = [B, 1 if is_convolution or is_instance else L_post, D_post]

            if post_condition_mask is None: post_condition_mask = tf.ones([B, L_post], dtype=dtype)
            if expanding or (n_batch_dims > 1):
                post_condition = tf.reshape(post_condition, condition_shaper)
                post_condition_mask = tf.reshape(post_condition_mask, [B, L_post])

            condition_goal = 'b' + broad_goal + 'p'
            condition_einsum_mask = tf.tile(post_condition_mask[Ellipsis, None], [1, 1, D_post])

            reciprocal_condition_mask_sum = tf.math.maximum(tf.einsum("blp->" + condition_goal, condition_einsum_mask), epsilon) ** (-1.0)
            post_condition = tf.einsum(",".join(["blp", "blp", condition_goal]) + "->" + condition_goal, post_condition, condition_einsum_mask, reciprocal_condition_mask_sum)
            post_condition = tf.reshape(post_condition, condition_reshaper)
            post_bool = instance_bool or is_convolution

            if is_group:
                weight_shape =  ([] if post_bool else [L_post]) + [D_post, G]
                weight_start = 'pg' if post_bool else 'lpg'
                bias_shape = [1 if post_bool else L_post, G, 1]
                post_reshaper = [B] + bias_shape
                post_goal = 'b' + broad_goal + 'g'
            else:
                weight_shape = ([] if post_bool else [L_post]) + ([D_post] if is_layer else [D_post, D_in])
                weight_start = ('' if post_bool else 'l') + ('p' if is_layer else 'pd')
                bias_shape = [1 if post_bool else L_post, 1 if is_layer else D_in]
                post_reshaper = [B] + bias_shape
                post_goal = 'b' + broad_goal + ('' if is_layer else 'd')

            if scale:
                post_gamma_weight = tf.get_variable("post-gamma_weight", shape=weight_shape, dtype=dtype, initializer=post_gamma_weight_initializer,
                        regularizer=post_gamma_weight_regularizer, trainable=trainable, constraint=post_gamma_weight_constraint)
                if post_gamma_weight_normalizer is not None: post_gamma_weight = post_gamma_weight_normalizer(post_gamma_weight)
                post_gamma = tf.reshape(tf.einsum(",".join(["blp", weight_start]) + "->" + post_goal, post_condition, post_gamma_weight), post_reshaper)
                if post_gamma_use_bias:
                    post_gamma_bias = tf.get_variable("post-gamma_bias", shape=bias_shape, dtype=dtype, initializer=post_gamma_bias_initializer,
                            regularizer=post_gamma_bias_regularizer, trainable=trainable, constraint=post_gamma_bias_constraint)
                    post_gamma = post_gamma + post_gamma_bias[None]
                if post_gamma_activation is not None: post_gamma = post_gamma_activation(post_gamma)
            if shift:
                post_beta_weight = tf.get_variable("post-beta_weight", shape=weight_shape, dtype=dtype, initializer=post_beta_weight_initializer,
                        regularizer=post_beta_weight_regularizer, trainable=trainable, constraint=post_beta_weight_constraint)
                if post_beta_weight_normalizer is not None: post_beta_weight = post_beta_weight_normalizer(post_beta_weight)
                post_beta = tf.reshape(tf.einsum(",".join(["blp", weight_start]) + "->" + post_goal, post_condition, post_beta_weight), post_reshaper)
                if post_beta_use_bias:
                    post_beta_bias = tf.get_variable("post-beta_bias", shape=bias_shape, dtype=dtype, initializer=post_beta_bias_initializer,
                            regularizer=post_beta_bias_regularizer, trainable=trainable, constraint=post_beta_bias_constraint)
                    post_beta = post_beta + post_beta_bias[None]
                if post_beta_activation is not None: post_beta = post_beta_activation(post_beta)

        # scale and shift
        if 'ADAPTIVE' in where:
            if scale: scaler = post_gamma
            if shift: shifter = post_beta
        else:
            if scale:
                gamma = tf.get_variable('gamma', shape=scale_shift_shape, dtype=dtype, initializer=gamma_initializer,
                        regularizer=gamma_regularizer, trainable=trainable, constraint=gamma_constraint)
                if gamma_normalizer: gamma = gamma_normalizer(gamma)
                scaler = gamma if post_condition is None else (gamma + post_gamma)
            if shift:
                beta = tf.get_variable('beta', shape=scale_shift_shape, dtype=dtype, initializer=beta_initializer,
                        regularizer=beta_regularizer, trainable=trainable, constraint=beta_constraint)
                shifter = beta if post_condition is None else (beta + post_beta)
        out_ = (out_ * scaler) + shifter

        # shape recovery
        out_ = tf.reshape(out_, in_shape)
        if activation: out_ = activation(out_)
        out_ = out_ * base_mask[Ellipsis, None]

    return out_
'''


def normalization(
        in_,
        where,
        mask=None,
        n_batch_dims=1,
        groups=1,
        epsilon=1e-8,
        momentum=0.99,  # for BN
        activation=None,
        scale=True,
        gamma_initializer=tf.ones_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        gamma_normalizer=None,
        shift=True,
        beta_initializer=tf.zeros_initializer(),
        beta_regularizer=None,
        beta_constraint=None,
        pre_condition=None,
        pre_groups=1,
        pre_use_bias=True,
        pre_activation=None,
        pre_weight_initializer=tf.glorot_uniform_initializer(),
        pre_weight_regularizer=None,
        pre_weight_constraint=None,
        pre_weight_normalizer=None,
        pre_bias_initializer=tf.zeros_initializer(),
        pre_bias_regularizer=None,
        pre_bias_constraint=None,
        post_condition=None,
        post_condition_mask=None,
        post_gamma_groups=1,
        post_gamma_use_bias=True,
        post_gamma_activation=None,
        post_gamma_weight_initializer=tf.glorot_uniform_initializer(),
        post_gamma_weight_regularizer=None,
        post_gamma_weight_constraint=None,
        post_gamma_weight_normalizer=None,
        post_gamma_bias_initializer=tf.zeros_initializer(),
        post_gamma_bias_regularizer=None,
        post_gamma_bias_constraint=None,
        post_beta_groups=1,
        post_beta_use_bias=True,
        post_beta_activation=None,
        post_beta_weight_initializer=tf.glorot_uniform_initializer(),
        post_beta_weight_regularizer=None,
        post_beta_weight_constraint=None,
        post_beta_weight_normalizer=None,
        post_beta_bias_initializer=tf.zeros_initializer(),
        post_beta_bias_regularizer=None,
        post_beta_bias_constraint=None,
        trainable=True,
        scope='normalization',
        # n_devices=1
):
    """
    where:
        batch -> batch normalization,
        convolution_batch -> batch normalization after convolution layer
        instance -> instance normalization
        layer -> layer normalization
        convolution_layer -> layer normalization after convolution layer
        group -> group normalization
        *** `adaptive` -> require post_condition, do not generate gamma and beta, only generate post-gamma and post-beta
        *** no `adaptive` but post_condition given -> conditional normalization
    """
    where = where.upper()
    if 'ADAPTIVE' in where and post_condition is None: raise ValueError(
        "'ADAPTIVE' normalization requires post_condition vertor.")
    is_batch = ('BATCH' in where)
    is_convolution = ('CONVOLUTION' in where)
    is_instance = ('INSTANCE' in where)
    is_group = ('GROUP' in where)
    is_layer = ('LAYER' in where)

    with tf.variable_scope(scope):
        # graph_name = tf.get_default_graph().get_name_scope()

        # shape information
        dtype = in_.dtype
        decay = 1.0 - momentum

        in_shape = shape_of(in_)
        B_axis = [axis for axis in range(n_batch_dims)]
        Bs = in_shape[:n_batch_dims]
        B = tf.reduce_prod(Bs)
        L = tf.reduce_prod(in_shape[n_batch_dims:-1])
        D_in = in_shape[-1]
        expanding = len(in_shape) - n_batch_dims > 2

        if mask == None: mask = tf.ones(in_shape[:-1], dtype=dtype)
        base_mask = mask

        if expanding or (n_batch_dims > 1):
            in_ = tf.reshape(in_, [B, L, D_in])
            mask = tf.reshape(mask, [B, L])

        if pre_condition is not None:
            pre_shape = shape_of(pre_condition)
            L_pre = tf.reduce_prod(pre_shape[n_batch_dims:-1])
            pre_condition = tf.reshape(pre_condition, [B, L_pre, pre_shape[-1]])
            pre_condition = tf.cond(tf.equal(L_pre, 1), lambda: tf.tile(pre_condition, [1, L, 1]),
                                    lambda: pre_condition)

            in_ = dense(tf.concat([in_, pre_condition], axis=-1) * mask[Ellipsis, None], in_shape[-1],
                        groups=pre_groups, use_bias=pre_use_bias, activation=pre_activation,
                        weight_initializer=pre_weight_initializer, weight_regularizer=pre_weight_regularizer,
                        weight_constraint=pre_weight_constraint, weight_normalizer=pre_weight_normalizer,
                        bias_initializer=pre_bias_initializer, bias_regularizer=pre_bias_regularizer,
                        bias_constraint=pre_bias_constraint, scope='pre_dense', trainable=trainable)

        if is_instance or is_convolution:
            broad_axis = [1]
            broad_goal = ''
        else:
            broad_axis = []
            broad_goal = 'l'

        if is_batch:
            axis = [0] + broad_axis
            goal = broad_goal
        else:
            axis = broad_axis
            goal = 'b' + broad_goal

        if is_group:
            goal = goal + 'g'
        elif not is_layer:
            goal = goal + 'd'

        instance_bool = is_instance or tf.is_tensor(L)
        if is_group:
            G = min(groups, D_in)
            D_G = D_in // G
            scale_shift_shape = [1, L if (is_convolution and not instance_bool) else 1, G, 1]
            reshaper = [1 if is_batch else B, 1 if (is_convolution or is_instance) else L, G, 1]
            einsum_mask = tf.tile(mask[Ellipsis, None, None], [1, 1, G, D_G])
            start = "blgd"
            in_ = tf.reshape(in_, [B, L, G, D_G])
        else:
            scale_shift_shape = [1, L if (is_convolution and not instance_bool) else 1, 1 if is_layer else D_in]
            reshaper = [1 if is_batch else B, 1 if (is_convolution or is_instance) else L, 1 if is_layer else D_in]
            einsum_mask = tf.tile(mask[Ellipsis, None], [1, 1, D_in])
            start = "bld"

        if is_batch:
            # recorded mean, variance
            moving_mean = tf.get_variable(
                'moving_mean', shape=scale_shift_shape, dtype=dtype, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', shape=scale_shift_shape, dtype=dtype, initializer=tf.ones_initializer,
                trainable=False)
            tf.add_to_collection("MOVING_AVERAGE", moving_mean)
            tf.add_to_collection("MOVING_AVERAGE", moving_variance)

            # mean, variance
            if trainable:
                if get_current_tower_context().is_main_training_tower:
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, moving_mean)
                    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, moving_variance)

                batch_string = f"{start},{start}->{goal}"
                assign_goal = re.sub("l", "", goal) if (tf.is_tensor(L) and not is_instance) else goal
                assign_shaper = reshaper[:1] + [1] + reshaper[2:]

                batch_sum = tf.einsum(batch_string, in_, einsum_mask)
                batch_sum_square = tf.einsum(batch_string, tf.math.square(in_), einsum_mask)
                batch_mask_sum = tf.einsum(start + "->" + goal, einsum_mask)
                batch_mask_reciprocal_sum = tf.where(tf.math.greater(batch_mask_sum, 0), tf.reciprocal(batch_mask_sum),
                                                     tf.zeros_like(batch_mask_sum))

                batch_mean = batch_sum * batch_mask_reciprocal_sum

                batch_mean_square = batch_sum_square * batch_mask_reciprocal_sum
                batch_variance = batch_mean_square - tf.math.square(batch_mean)
                batch_variance = batch_variance * batch_mask_sum * tf.where(tf.math.greater(batch_mask_sum, 1),
                                                                            tf.reciprocal(batch_mask_sum - 1),
                                                                            tf.zeros_like(batch_mask_sum))

                if tf.is_tensor(L) and not is_instance:
                    assign_mask_sum = tf.einsum(f"{goal}->{assign_goal}", batch_mask_sum)
                    assign_mask_reciprocal_sum = tf.where(tf.math.greater(assign_mask_sum, 0),
                                                          tf.reciprocal(assign_mask_sum), tf.zeros_like(batch_mask_sum))

                    assign_mean = tf.einsum(f"{goal}->{assign_goal}", batch_sum) * assign_mask_reciprocal_sum

                    assign_mean_square = tf.einsum(f"{goal}->{assign_goal}",
                                                   batch_sum_square) * assign_mask_reciprocal_sum
                    assign_variance = assign_mean_square - tf.math.square(assign_mean)
                    assign_variance = assign_variance * assign_mask_sum * tf.where(tf.math.greater(assign_mask_sum, 1),
                                                                                   tf.reciprocal(assign_mask_sum - 1),
                                                                                   tf.zeros_like(batch_mask_sum))

                    mean = (decay * tf.cast(moving_mean, dtype)) + ((1.0 - decay) * tf.reshape(batch_mean, reshaper))
                    variance = (decay * tf.cast(moving_variance, dtype)) + (
                            (1.0 - decay) * tf.reshape(batch_variance, reshaper))

                else:
                    assign_mean = batch_mean
                    assign_variance = batch_variance

                from tensorflow.python.training import moving_averages
                with tf.control_dependencies([
                    moving_averages.assign_moving_average(moving_mean, assign_mean, decay, zero_debias=False,
                                                          name='mean_ema_op'),
                    moving_averages.assign_moving_average(moving_variance, assign_variance, decay, zero_debias=False,
                                                          name='variance_ema_op'),
                ]):
                    if tf.is_tensor(L) and not is_instance:
                        out_ = (in_ - mean) * tf.math.rsqrt(tf.math.maximum(variance, epsilon))
                    else:
                        out_ = (in_ - moving_mean) * tf.math.rsqrt(tf.math.maximum(moving_variance, epsilon))

            else:
                out_ = (in_ - moving_mean) * tf.math.rsqrt(tf.math.maximum(moving_variance, epsilon))

        else:
            mask_reciprocal_sum = tf.math.maximum(tf.einsum(start + "->" + goal, einsum_mask), epsilon) ** (-1.0)
            mean = tf.reshape(
                tf.einsum(",".join([start, start, goal]) + "->" + goal, in_, einsum_mask, mask_reciprocal_sum),
                reshaper)
            delta = in_ - mean
            variance = tf.reshape(
                tf.einsum(",".join([start, start, goal]) + "->" + goal, tf.math.square(delta), einsum_mask,
                          mask_reciprocal_sum), reshaper)

            out_ = delta * tf.math.rsqrt(tf.math.maximum(variance, epsilon))

        if post_condition is not None:
            post_shape = shape_of(post_condition)
            L_post = tf.reduce_prod(post_shape[n_batch_dims:-1])
            D_post = post_shape[-1]
            condition_shaper = [B, L_post, D_post]
            condition_reshaper = [B, 1 if is_convolution or is_instance else L_post, D_post]

            if post_condition_mask is None: post_condition_mask = tf.ones([B, L_post], dtype=dtype)
            if expanding or (n_batch_dims > 1):
                post_condition = tf.reshape(post_condition, condition_shaper)
                post_condition_mask = tf.reshape(post_condition_mask, [B, L_post])

            condition_goal = 'b' + broad_goal + 'p'
            condition_einsum_mask = tf.tile(post_condition_mask[Ellipsis, None], [1, 1, D_post])

            reciprocal_condition_mask_sum = tf.math.maximum(tf.einsum("blp->" + condition_goal, condition_einsum_mask),
                                                            epsilon) ** (-1.0)
            post_condition = tf.einsum(",".join(["blp", "blp", condition_goal]) + "->" + condition_goal, post_condition,
                                       condition_einsum_mask, reciprocal_condition_mask_sum)
            post_condition = tf.reshape(post_condition, condition_reshaper)
            post_bool = instance_bool or is_convolution

            if is_group:
                weight_shape = ([] if post_bool else [L_post]) + [D_post, G]
                weight_start = 'pg' if post_bool else 'lpg'
                bias_shape = [1 if post_bool else L_post, G, 1]
                post_reshaper = [B] + bias_shape
                post_goal = 'b' + broad_goal + 'g'
            else:
                weight_shape = ([] if post_bool else [L_post]) + ([D_post] if is_layer else [D_post, D_in])
                weight_start = ('' if post_bool else 'l') + ('p' if is_layer else 'pd')
                bias_shape = [1 if post_bool else L_post, 1 if is_layer else D_in]
                post_reshaper = [B] + bias_shape
                post_goal = 'b' + broad_goal + ('' if is_layer else 'd')

            if scale:
                post_gamma_weight = tf.get_variable("post-gamma_weight", shape=weight_shape, dtype=dtype,
                                                    initializer=post_gamma_weight_initializer,
                                                    regularizer=post_gamma_weight_regularizer, trainable=trainable,
                                                    constraint=post_gamma_weight_constraint)
                if post_gamma_weight_normalizer is not None: post_gamma_weight = post_gamma_weight_normalizer(
                    post_gamma_weight)
                post_gamma = tf.reshape(
                    tf.einsum(",".join(["blp", weight_start]) + "->" + post_goal, post_condition, post_gamma_weight),
                    post_reshaper)
                if post_gamma_use_bias:
                    post_gamma_bias = tf.get_variable("post-gamma_bias", shape=bias_shape, dtype=dtype,
                                                      initializer=post_gamma_bias_initializer,
                                                      regularizer=post_gamma_bias_regularizer, trainable=trainable,
                                                      constraint=post_gamma_bias_constraint)
                    post_gamma = post_gamma + post_gamma_bias[None]
                if post_gamma_activation is not None: post_gamma = post_gamma_activation(post_gamma)
            if shift:
                post_beta_weight = tf.get_variable("post-beta_weight", shape=weight_shape, dtype=dtype,
                                                   initializer=post_beta_weight_initializer,
                                                   regularizer=post_beta_weight_regularizer, trainable=trainable,
                                                   constraint=post_beta_weight_constraint)
                if post_beta_weight_normalizer is not None: post_beta_weight = post_beta_weight_normalizer(
                    post_beta_weight)
                post_beta = tf.reshape(
                    tf.einsum(",".join(["blp", weight_start]) + "->" + post_goal, post_condition, post_beta_weight),
                    post_reshaper)
                if post_beta_use_bias:
                    post_beta_bias = tf.get_variable("post-beta_bias", shape=bias_shape, dtype=dtype,
                                                     initializer=post_beta_bias_initializer,
                                                     regularizer=post_beta_bias_regularizer, trainable=trainable,
                                                     constraint=post_beta_bias_constraint)
                    post_beta = post_beta + post_beta_bias[None]
                if post_beta_activation is not None: post_beta = post_beta_activation(post_beta)

        # scale and shift
        if 'ADAPTIVE' in where:
            if scale: scaler = post_gamma
            if shift: shifter = post_beta
        else:
            if scale:
                gamma = tf.get_variable('gamma', shape=scale_shift_shape, dtype=dtype, initializer=gamma_initializer,
                                        regularizer=gamma_regularizer, trainable=trainable, constraint=gamma_constraint)
                if gamma_normalizer: gamma = gamma_normalizer(gamma)
                scaler = gamma if post_condition is None else (gamma + post_gamma)
            if shift:
                beta = tf.get_variable('beta', shape=scale_shift_shape, dtype=dtype, initializer=beta_initializer,
                                       regularizer=beta_regularizer, trainable=trainable, constraint=beta_constraint)
                shifter = beta if post_condition is None else (beta + post_beta)
        out_ = (out_ * scaler) + shifter

        # shape recovery
        out_ = tf.reshape(out_, in_shape)
        if activation: out_ = activation(out_)
        out_ = out_ * base_mask[Ellipsis, None]

    return out_


def dense(
        in_,
        units,
        mask=None,
        groups=1,
        use_bias=True,
        activation=None,
        weight_initializer=tf.glorot_uniform_initializer(),
        weight_regularizer=None,
        weight_constraint=None,
        weight_normalizer=None,
        bias_initializer=tf.zeros_initializer(),
        bias_regularizer=None,
        bias_constraint=None,
        trainable=True,
        scope='dense'
):
    assert isinstance(groups, int), "'groups' must be integer"
    assert groups > 0

    with tf.variable_scope(scope):
        # graph_name = tf.get_default_graph().get_name_scope()

        dtype = in_.dtype
        in_shape = shape_of(in_)
        is_group = groups > 1

        if mask is not None: in_ = in_ * mask[Ellipsis, None]

        if is_group:
            in_ = tf.reshape(in_, in_shape[:-1] + [in_shape[-1] // groups, groups])
            var_shape = [in_shape[-1] // groups, groups, units // groups]
            string_equation = "...dg,dgu->...gu"
        else:
            var_shape = [in_shape[-1], units]
            string_equation = "...d,du->...u"

        weight = tf.get_variable("weight", shape=var_shape, dtype=dtype, initializer=weight_initializer,
                                 regularizer=weight_regularizer, trainable=trainable, constraint=weight_constraint)
        if weight_normalizer: weight = weight_normalizer(weight)
        print(var_shape, shape_of(weight), scope)
        out_ = tf.einsum(string_equation, in_, weight)
        if is_group: out_ = tf.reshape(out_, in_shape[:-1] + [units])

        if use_bias:
            bias = tf.get_variable("bias", shape=([1] * (len(in_shape) - 1) + [units]), dtype=dtype,
                                   initializer=bias_initializer, regularizer=bias_regularizer, trainable=trainable,
                                   constraint=bias_constraint)
            if mask is not None: bias = bias * mask[Ellipsis, None]
            out_ = out_ + bias

        if activation: out_ = activation(out_)
    return out_


def modulated_dense(
        in_,
        units,
        mask=None,
        groups=1,
        use_bias=True,
        activation=None,
        weight_initializer=tf.glorot_uniform_initializer(),
        weight_regularizer=None,
        weight_constraint=None,
        weight_normalizer=None,
        bias_initializer=tf.zeros_initializer(),
        bias_regularizer=None,
        bias_constraint=None,
        trainable=True,
        scope='dense'
):
    assert isinstance(groups, int), "'groups' must be integer"
    assert groups > 0

    with tf.variable_scope(scope):
        # graph_name = tf.get_default_graph().get_name_scope()

        dtype = in_.dtype
        in_shape = shape_of(in_)
        is_group = groups > 1

        if mask is not None: in_ = in_ * mask[Ellipsis, None]

        if is_group:
            in_ = tf.reshape(in_, in_shape[:-1] + [in_shape[-1] // groups, groups])
            var_shape = [in_shape[-1] // groups, groups, units // groups]
            string_equation = "...dg,dgu->...gu"
            modulate_string = "dgu,dgu->gu"
        else:
            var_shape = [in_shape[-1], units]
            string_equation = "...d,du->...u"
            modulate_string = "du,du->u"

        weight = tf.get_variable("weight", shape=var_shape, dtype=dtype, initializer=weight_initializer,
                                 regularizer=weight_regularizer, trainable=trainable, constraint=weight_constraint)
        if weight_normalizer: weight = weight_normalizer(weight)
        weight *= tf.math.rsqrt(tf.einsum(modulate_string, weight, weight) + 1e-16)[None]

        out_ = tf.einsum(string_equation, in_, weight)
        if is_group: out_ = tf.reshape(out_, in_shape[:-1] + [units])

        if use_bias:
            bias = tf.get_variable("bias", shape=([1] * (len(in_shape) - 1) + [units]), dtype=dtype,
                                   initializer=bias_initializer, regularizer=bias_regularizer, trainable=trainable,
                                   constraint=bias_constraint)
            if mask is not None: bias = bias * mask[Ellipsis, None]
            out_ = out_ + bias

        if activation: out_ = activation(out_)
    return out_


def dense_einsum(in_,
                 out_shape,
                 n_summed_dims=1,
                 mask=None,
                 use_bias=True,
                 activation=None,
                 weight_initializer=tf.glorot_uniform_initializer(),
                 weight_regularizer=None,
                 weight_constraint=None,
                 weight_normalizer=None,
                 bias_initializer=tf.zeros_initializer(),
                 bias_regularizer=None,
                 bias_constraint=None,
                 trainable=True,
                 scope='dense_einsum'):
    with tf.variable_scope(scope):
        dtype = in_.dtype
        in_shape = shape_of(in_)
        in_rank = len(in_shape)
        n_free_dims = in_rank - n_summed_dims
        out_rank = len(out_shape)

        if mask is not None: in_ = in_ * mask[Ellipsis, None]

        _CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
        input_str = ""
        weight_str = ""
        output_str = ""
        letter_offset = 0
        for i in range(n_free_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char
            output_str += char

        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = _CHR_IDX[i + letter_offset]
            weight_str += char
            output_str += char

        letter_offset += bound_dims
        for i in range(output_dims):
            char = _CHR_IDX[i + letter_offset]
            weight_str += char
            output_str += char

        string_equation = input_str + "," + weight_str + "->" + output_str

        weight_shape = in_shape[n_free_dims:] + output_shape
        weight = tf.get_variable("weight", shape=weight_shape, dtype=dtype, initializer=weight_initializer,
                                 regularizer=weight_regularizer, trainable=trainable, constraint=weight_constraint)
        if weight_normalizer is not None: weight = weight_normalizer(weight)
        out_ = tf.einsum(string_equation, in_, weight)

        if use_bias:
            bias = tf.get_variable("bias", shape=output_shape, dtype=dtype, intializer=bias_initializer,
                                   regularizer=bias_regularizer, trainable=trainable, constraint=bias_constraint)
            if mask is not None: bias = bias * mask[Ellipsis, None]
            out_ = out_ + bias

        if activation is not None:
            out_ = activation(out_)

    return out_


def convolution(in_,
                rank,
                filters,
                kernel_size,
                strides=1,
                dilations=1,
                padding='valid',
                mask_in=None,
                mask_out=None,
                groups=1,
                use_bias=True,
                activation=None,
                weight_initializer=tf.glorot_uniform_initializer(),
                weight_regularizer=None,
                weight_constraint=None,
                weight_normalizer=None,
                bias_initializer=tf.zeros_initializer(),
                bias_regularizer=None,
                bias_constraint=None,
                data_format=None,
                trainable=True,
                scope='convolution'):
    """
    convolution of <rank>-dimensional
    """
    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        # rank check
        if rank not in {1, 2, 3}:
            raise ValueError('The number of spatial dimensions must be one of 1, 2 or 3 but saw {}.'.format(rank))

        # filters check
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters % groups != 0:
            raise ValueError('The number of filters must be evenly divisible by the number of groups.'
                             'Received: groups={}, filters={}.'.format(groups, filters))

        # padding check
        padding = padding.upper()
        if padding == 'CAUSAL' and rank != 1:
            raise ValueError('Causal padding is only supported for 1-dimensional convolution')

        # channels check
        dtype = in_.dtype
        in_shape = shape_of(in_)
        in_units = in_shape[-1]
        if in_units % groups != 0:
            raise ValueError('The number of input channels must be evenly divisible by the number of groups.'
                             'Received groups={}, but the input has {} channels (full input shape is {}).'.format(
                groups, in_units, in_shape))

        # kernel_size control
        if isinstance(kernel_size, int): kernel_size = [kernel_size, ] * rank
        kernel_size = list(kernel_size)
        if len(kernel_size) != rank:
            raise ValueError('The `kernel_size` argument must be a list of {} integers.'
                             'Received: {}.'.format(rank, kernel_size))
        for single_size in kernel_size:
            assert isinstance(single_size, int)
        if not all(kernel_size):
            raise ValueError('`kernel_size` cannot contain 0(s).'
                             'Received: {}'.format(kernel_size))

        # internal convolution operation
        n_total_dims = len(in_shape)
        n_batch_dims = n_total_dims - rank - 1
        batch_dims = list(range(0, n_batch_dims))

        # mask
        if mask_in is not None: in_ = in_ * mask_in[Ellipsis, None]
        # with tf.control_dependencies([tf_print(f'{graph_name}/in_*mask', in_)]): in_ = tf.identity(in_)

        # causal mask
        if padding == 'CAUSAL':
            in_ = tf.pad(in_, causal_padding(in_, kernel_size, dilations, data_format=data_format))
            padding = 'VALID'

        weight_shape = kernel_size + [in_units // groups, filters]
        weight = tf.get_variable(
            'weight', shape=weight_shape, dtype=dtype, initializer=weight_initializer, regularizer=weight_regularizer,
            trainable=trainable, constraint=weight_constraint)
        if weight_normalizer: weight = weight_normalizer(weight)
        if groups > 1: weight = tf.tile(weight, [1] * rank + [groups, 1])

        # manufacture shape
        """
        if groups > 1:
            if data_format == 'channels_first':
                in_ = tf.reshape(in_, in_shape[:n_batch_dims] + [groups, in_units//groups] + in_shape[n_batch_dims+1:])
                in_ = tf.transpose(in_, batch_dims + [n_batch_dims] + list(range(n_batch_dims + 2, n_total_dims+1)) + [n_batch_dims + 1])
                in_ = tf.reshape(in_, [-1] + in_shape[n_batch_dims+1:] + [in_units//groups])
            else:
                in_ = tf.reshape(in_, in_shape[:-1] + [groups, in_units//groups])
                in_ = tf.transpose(in_, batch_dims + [n_total_dims-1] + list(range(n_batch_dims, n_total_dims-1)) + [n_total_dims])
                in_ = tf.reshape(in_, [-1] + in_shape[n_batch_dims:n_total_dims] + [in_units//groups])
        else:
        """
        in_ = tf.reshape(in_, [-1] + in_shape[n_batch_dims:])
        if data_format == 'channels_first': in_ = tf.transpose(in_, [0] + list(range(2, rank + 2)) + [1])
        # with tf.control_dependencies([tf_print(f'{graph_name}/in_transformed', in_)]): in_ = tf.identity(in_)

        # strides
        if strides is None:
            strides = [1] * (rank + 2)
        else:
            if isinstance(strides, int): strides = [strides, ] * rank
            strides = list(strides)
            for single_size in strides:
                assert isinstance(single_size, int)
            if not all(strides):
                raise ValueError('`strides` cannot contain 0(s).'
                                 'Received: {}'.format(strides))
            n_stride_dims = len(strides)
            if n_stride_dims != (rank + 2):
                if n_stride_dims == 1:
                    strides = strides * rank
                elif n_stride_dims != rank:
                    raise ValueError(
                        "strides should be of length 1, {} or {} but was {}.".format(rank, n_total_dims, n_stride_dims))
                strides = [1] + strides + [1]

        # dilations
        if dilations is None:
            dilations = [1] * (rank + 2)
        else:
            if isinstance(dilations, int): dilations = [dilations, ] * rank
            dilations = list(dilations)
            for single_size in dilations:
                assert isinstance(single_size, int)
            if not all(dilations):
                raise ValueError('`dilations` cannot contain 0(s).'
                                 'Received: {}'.format(dilations))
            n_dilation_dims = len(dilations)
            if n_dilation_dims != (rank + 2):
                if n_dilation_dims == 1:
                    dilations = dilations * rank
                elif n_dilation_dims != rank:
                    raise ValueError("dilations should be of length 1, {} or {} but was {}.".format(rank, n_total_dims,
                                                                                                    n_dilation_dims))
                dilations = [1] + dilations + [1]

        # selection
        if rank == 3:
            convolution_performer = gen_nn_ops.conv3d
        else:
            if rank == 1:
                in_ = tf.expand_dims(in_, axis=1)
                weight = weight[None, Ellipsis]
                strides = [strides[0], 1] + strides[1:]
                dilations = [dilations[0], 1] + dilations[1:]
            convolution_performer = gen_nn_ops.conv2d

        # perform operation
        out_ = convolution_performer(in_, weight, strides, padding, dilations=dilations, name="Conv{}D".format(rank))
        if rank == 1:
            out_ = tf.squeeze(out_, axis=[1])
        # with tf.control_dependencies([tf_print(f'{graph_name}/conv_direct_out', out_)]): out_ = tf.identity(out_)

        # bias
        if use_bias:
            bias = tf.get_variable(
                'bias', shape=([1] * (rank + 1) + [filters]), dtype=dtype, initializer=bias_initializer,
                regularizer=bias_regularizer,
                trainable=trainable, constraint=bias_constraint)
            # with tf.control_dependencies([tf_print(f'{graph_name}/bias', bias)]): bias = tf.identity(bias)
            if mask_out is not None: bias = bias * mask_out[Ellipsis, None]
            out_ += bias
            # with tf.control_dependencies([tf_print(f'{graph_name}/bias_added_out', out_)]): out_ = tf.identity(out_)

        # recover shape
        out_shape = shape_of(out_)

        if data_format == 'channels_first':
            out_ = tf.transpose(out_, [0, rank + 1] + list(range(1, rank + 1)))
            out_ = tf.reshape(out_, in_shape[:n_batch_dims] + out_shape[-1:] + out_shape[1:-1])
        else:
            out_ = tf.reshape(out_, in_shape[:n_batch_dims] + out_shape[1:])

        if activation: out_ = activation(out_)
        # with tf.control_dependencies([tf_print(f'{graph_name}/conv_total_out', out_)]): out_ = tf.identity(out_)
    return out_


def attention(queries,
              keys,
              values,
              input_relation='self',
              heads=4,
              units=None,
              score_units=None,
              mask_query=None,
              mask_key=None,
              mask_value=None,
              cache=None,
              use_positional_encoding=False,
              activation=None,
              projector='convolution',
              projector_kernel_size=3,
              projector_strides=1,
              projector_dilations=None,
              projector_mask_query=None,
              projector_mask_key=None,
              projector_mask_value=None,
              projector_groups=1,
              projector_use_bias=True,
              projector_activation=None,
              projector_weight_normalizer=None,
              projector_weight_initializer=tf.glorot_uniform_initializer(),
              projector_weight_regularizer=None,
              projector_weight_constraint=None,
              projector_bias_initializer=tf.zeros_initializer(),
              projector_bias_regularizer=None,
              projector_bias_constraint=None,
              trainable=True,
              scope='local_attention'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))

    *Bs_q, L_out, D_q = shape_of(queries)
    *Bs, L, D_k = shape_of(keys)
    *Bs_v, L_v, D_v = shape_of(values)

    # assert Bs_q == Bs and Bs_v == Bs, f"Batch of queries : {Bs_q}  Batch of keys : {Bs}  Batch of values : {Bs_v}"
    # assert L_v == L

    # n_B_axis = len(Bs)
    # B_axis = [axis for axis in range(n_B_axis)]

    D_qk = max(D_q, D_k)

    H = heads if isinstance(heads, int) else 1

    D_out = units if isinstance(units, int) else D_q
    if D_out % H != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(H, D_out))

    D_score = score_units if isinstance(score_units, int) else D_qk
    if D_score % H != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            H, D_score))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    #if not score_activation in [tf.nn.softmax]: raise ValueError(
    # "given method {} of score activation function is unsupported.".format(score_activation))

    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype
        B = tf.reduce_prod(Bs, keepdims=False)

        # batch shape change
        queries, keys, values = [tf.reshape(in_, [B, length, numerics]) for in_, length, numerics in
                                 [(queries, L_out, D_q), (keys, L, D_k), (values, L, D_v)]]

        mask_query = tf.ones([B, L_out], dtype=dtype) if mask_query is None else tf.reshape(mask_query, [B, L_out])
        mask_key = tf.ones([B, L], dtype=dtype) if mask_key is None else tf.reshape(mask_key, [B, L])
        mask_value = tf.ones([B, L], dtype=dtype) if mask_value is None else tf.reshape(mask_value, [B, L])

        if projector_mask_query is not None: projector_mask_query = tf.reshape(projector_mask_query, [B, L_out])
        if projector_mask_key is not None: projector_mask_key = tf.reshape(projector_mask_key, [B, L])
        if projector_mask_value is not None: projector_mask_value = tf.reshape(projector_mask_value, [B, L])

        def material_projection(xs, units, material_units, length, mask, name='X'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = tf.reshape(
                    convolution(xs, 1, material_units, projector_kernel_size, projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'dense':
                X = tf.reshape(
                    dense(xs, material_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias,
                          activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    weight = tf.get_variable('weight', shape=[units, H, material_units // H], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: weight = projector_weight_normalizer(weight)
                    X = tf.einsum("blc,chd->blhd", xs, weight)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[material_units // H], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None, None]
            else:
                raise NotImplementedError
            return X

        # material projection
        Q = material_projection(queries, D_q, D_score, L_out, projector_mask_query, 'Q')
        K = material_projection(keys, D_k, D_score, L, projector_mask_key, 'K')
        V = material_projection(values, D_v, D_out, L, projector_mask_value, 'V')

        # cache
        if cache:
            K, V = [tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)]
            cache_shape = shape_of(cache['K'])
            L = cache_shape[-2] + L
            mask_key = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_key], axis=-2)
            mask_value = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_value], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'

        score = tf.einsum('bshd,blhd->bslh', Q, K)
        score = tf.nn.softmax(score, axis=-2)
        out_ = tf.einsum('bslh,blhd->bshd', score, K)

        def restore_projection(xs, units, restore_units, length, mask, name='restore'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = convolution(tf.reshape(xs, [B, length, units]), 1, restore_units, projector_kernel_size,
                                projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name)
            elif projector == 'dense':
                X = dense(tf.reshape(xs, [B, length, units]), restore_units, mask=mask, groups=projector_groups,
                          use_bias=projector_use_bias, activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name)
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    weight = tf.get_variable('weight', shape=[H, units // H, restore_units], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: weight = projector_weight_normalizer(weight)
                    X = tf.einsum("blho,hor->blr", xs, weight)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[restore_units], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None]
            else:
                raise NotImplementedError
            return X

        out_ = restore_projection(out_, D_out, D_out, L_out, mask_query, name='restore')
        out_ = tf.reshape(out_, Bs_q + [L_out, D_out])
        if activation: out_ = activation(out_)

    # return out_, cache, score
    return out_


def kernelized_attention(queries,
                         keys,
                         values,
                         input_relation='self',
                         heads=4,
                         units=None,
                         kernel_size=7,
                         score_units=None,
                         mask_query=None,
                         mask_key=None,
                         mask_value=None,
                         cache=None,
                         use_positional_encoding=False,
                         activation=None,
                         projector='convolution',
                         projector_kernel_size=3,
                         projector_strides=1,
                         projector_dilations=None,
                         projector_mask_query=None,
                         projector_mask_key=None,
                         projector_mask_value=None,
                         projector_groups=1,
                         projector_use_bias=True,
                         projector_activation=None,
                         projector_weight_normalizer=None,
                         projector_weight_initializer=tf.glorot_uniform_initializer(),
                         projector_weight_regularizer=None,
                         projector_weight_constraint=None,
                         projector_bias_initializer=tf.zeros_initializer(),
                         projector_bias_regularizer=None,
                         projector_bias_constraint=None,
                         trainable=True,
                         scope='local_attention'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))

    *Bs_q, L_out, D_q = shape_of(queries)
    *Bs, L, D_k = shape_of(keys)
    *Bs_v, L_v, D_v = shape_of(values)

    # assert Bs_q == Bs and Bs_v == Bs, f"Batch of queries : {Bs_q}  Batch of keys : {Bs}  Batch of values : {Bs_v}"
    # assert L_v == L

    # n_B_axis = len(Bs)
    # B_axis = [axis for axis in range(n_B_axis)]

    D_qk = max(D_q, D_k)

    H = heads if isinstance(heads, int) else 1

    D_out = units if isinstance(units, int) else D_q
    if D_out % H != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(H, D_out))

    D_score = score_units if isinstance(score_units, int) else D_qk
    if D_score % H != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            H, D_score))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    #if not score_activation in [tf.nn.softmax]: raise ValueError(
    # "given method {} of score activation function is unsupported.".format(score_activation))

    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype
        B = tf.reduce_prod(Bs, keepdims=False)

        # batch shape change
        queries, keys, values = [tf.reshape(in_, [B, length, numerics]) for in_, length, numerics in
                                 [(queries, L_out, D_q), (keys, L, D_k), (values, L, D_v)]]

        mask_query = tf.ones([B, L_out], dtype=dtype) if mask_query is None else tf.reshape(mask_query, [B, L_out])
        mask_key = tf.ones([B, L], dtype=dtype) if mask_key is None else tf.reshape(mask_key, [B, L])
        mask_value = tf.ones([B, L], dtype=dtype) if mask_value is None else tf.reshape(mask_value, [B, L])

        if projector_mask_query is not None: projector_mask_query = tf.reshape(projector_mask_query, [B, L_out])
        if projector_mask_key is not None: projector_mask_key = tf.reshape(projector_mask_key, [B, L])
        if projector_mask_value is not None: projector_mask_value = tf.reshape(projector_mask_value, [B, L])

        def material_projection(xs, units, material_units, length, mask, name='X'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = tf.reshape(
                    convolution(xs, 1, material_units, projector_kernel_size, projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'dense':
                X = tf.reshape(
                    dense(xs, material_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias,
                          activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    weight = tf.get_variable('weight', shape=[units, H, material_units // H], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: weight = projector_weight_normalizer(weight)
                    X = tf.einsum("blc,chd->blhd", xs, weight)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[material_units // H], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None, None]
            else:
                raise NotImplementedError
            return X

        # material projection
        Q = material_projection(queries, D_q, D_score, L_out, projector_mask_query, 'Q')
        K = material_projection(keys, D_k, D_score, L, projector_mask_key, 'K')
        V = material_projection(values, D_v, D_out, L, projector_mask_value, 'V')

        # cache
        if cache:
            K, V = [tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)]
            cache_shape = shape_of(cache['K'])
            L = cache_shape[-2] + L
            mask_key = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_key], axis=-2)
            mask_value = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_value], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'

        half_size = kernel_size // 2
        gather_size = half_size + 1 if input_relation == 'causal' else kernel_size

        indices = ((L - 1) * tf.ones([1], dtype) if cache else tf.range(L))[:, None] + (tf.range(gather_size)[None])

        K = tf.gather(tf.pad(K, [[0, 0], [half_size, half_size], [0, 0], [0, 0]]), indices, axis=1)
        K += tf.get_variable("pos_enc_K", [1, 1, gather_size, 1, D_score // heads], trainable=trainable)
        K = K * mask_key[Ellipsis, None, None, None]

        V = tf.gather(tf.pad(V, [[0, 0], [half_size, half_size], [0, 0], [0, 0]]), indices, axis=1)
        V += tf.get_variable("pos_enc_V", [1, 1, gather_size, 1, D_out // heads], trainable=trainable)
        V = V * mask_value[Ellipsis, None, None, None]

        score = tf.einsum('blhd,blkhd->blkh', Q * mask_query[Ellipsis, None, None], K) * tf.math.rsqrt(
            tf.to_float(D_score))
        score_size = tf.tile(indices[None, Ellipsis, None], [B, 1, 1, H])
        key_len = tf.to_int32(tf.einsum('bl->b', mask_key))
        attention_mask = tf.logical_and(score_size >= half_size, score_size < half_size + key_len[:, None, None, None])

        score = tf.where(attention_mask, score, tf.ones_like(score) * (-2 ** 32 + 1))
        score = tf.nn.softmax(score, axis=-2)

        out_ = tf.einsum('blkh,blkhd->blhd', score, V)

        def restore_projection(xs, units, restore_units, length, mask, name='restore'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = convolution(tf.reshape(xs, [B, length, units]), 1, restore_units, projector_kernel_size,
                                projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name)
            elif projector == 'dense':
                X = dense(tf.reshape(xs, [B, length, units]), restore_units, mask=mask, groups=projector_groups,
                          use_bias=projector_use_bias, activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name)
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    weight = tf.get_variable('weight', shape=[H, units // H, restore_units], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: weight = projector_weight_normalizer(weight)
                    X = tf.einsum("blho,hor->blr", xs, weight)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[restore_units], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None]
            else:
                raise NotImplementedError
            return X

        out_ = restore_projection(out_, D_out, D_out, L_out, mask_query, name='restore')
        out_ = tf.reshape(out_, Bs_q + [L_out, D_out])

    # return out_, cache, score
    return out_


def modulated_kernel_attention(queries,
                               keys,
                               values,
                               style=None,
                               units=None,
                               score_units=None,
                               kernel_size=7,
                               heads=4,
                               input_relation='self',
                               demodulate=True,
                               fused=True,
                               mask_query=None,
                               mask_key=None,
                               mask_value=None,
                               cache=None,
                               use_positional_encoding=False,
                               activation=None,
                               score_activation=partial(tf.nn.softmax, axis=-2),
                               projector='einsum_dense',
                               projector_kernel_size=3,
                               projector_strides=1,
                               projector_dilations=None,
                               projector_mask_query=None,
                               projector_mask_key=None,
                               projector_mask_value=None,
                               projector_groups=1,
                               projector_use_bias=True,
                               projector_activation=None,
                               projector_weight_normalizer=None,
                               projector_gain=1,
                               projector_use_wscale=True,
                               projector_lrmul=1,
                               trainable=True,
                               scope='kernel_attention'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))

    *Bs_q, L_out, D_q = shape_of(queries)
    *Bs, L, D_k = shape_of(keys)
    *Bs_v, L_v, D_v = shape_of(values)

    D_qk = max(D_q, D_k)

    H = heads if isinstance(heads, int) else 1

    D_out = units if isinstance(units, int) else D_q
    if D_out % H != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(H, D_out))

    D_score = score_units if isinstance(score_units, int) else D_qk
    if D_score % H != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            H, D_score))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    #if not score_activation in [tf.nn.softmax]: raise ValueError(
    # "given method {} of score activation function is unsupported.".format(score_activation))

    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype
        B = tf.reduce_prod(Bs, keepdims=False)

        # batch shape change
        queries, keys, values = [tf.reshape(in_, [B, length, numerics]) for in_, length, numerics in
                                 [(queries, L_out, D_q), (keys, L, D_k), (values, L, D_v)]]

        mask_query = tf.ones([B, L_out], dtype=dtype) if mask_query is None else tf.reshape(mask_query, [B, L_out])
        mask_key = tf.ones([B, L], dtype=dtype) if mask_key is None else tf.reshape(mask_key, [B, L])
        mask_value = tf.ones([B, L], dtype=dtype) if mask_value is None else tf.reshape(mask_value, [B, L])

        if projector_mask_query is not None: projector_mask_query = tf.reshape(projector_mask_query, [B, L_out])
        if projector_mask_key is not None: projector_mask_key = tf.reshape(projector_mask_key, [B, L])
        if projector_mask_value is not None: projector_mask_value = tf.reshape(projector_mask_value, [B, L])

        '''
        def material_projection(xs, units, material_units, length, mask, name='X'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size, projector_dilations))
                X = tf.reshape(convolution(xs, 1, material_units, projector_kernel_size, projector_strides, projector_dilations, padding=('valid' if input_relation=='causal' else 'same'),
                    mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_initializer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name),
                    [B, length, H, material_units//H])
            elif projector == 'dense':
                X = tf.reshape(dense(xs, material_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name),
                    [B, length, H, material_units//H])
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_'+name):
                    weight = tf.get_variable('weight', shape=[units, H, material_units//H], dtype=dtype,
                            initializer=projector_weight_initializer, regularizer=projector_weight_regularizer, trainable=trainable, constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: weight = projector_weight_normalizer(weight)
                    X = tf.einsum("blc,chd->blhd", xs, weight)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[material_units//H], dtype=dtype,
                                initializer=projector_bias_initializer, regularizer=projector_bias_regularizer, trainable=trainable, constraint=projector_bias_constraint)
                        X = X + bias[None, None, None]
            else: raise NotImplementedError
            return X
        '''

        # TODO Further update: move modulation and demodulation into weight normalization level in operations.py
        def get_weight(shape, gain=1, use_wscale=True, lrmul=1, trainable=True, weight_name='weight'):
            fan_in = np.prod(shape[:-1], dtype=np.float32)
            he_std = gain / np.sqrt(fan_in)
            if use_wscale:
                init_std = 1.0 / lrmul
                runtime_coef = he_std * lrmul
            else:
                init_std = he_std / lrmul
                runtime_coef = lrmul
            return tf.get_variable(weight_name, shape=shape, initializer=tf.random_normal_initializer(0, init_std),
                                   trainable=trainable) * runtime_coef

        def modulated_projection(xs, style, units, material_units, length, mask, name='X'):

            with tf.variable_scope(name):
                if style is not None:
                    style_shape = shape_of(style)
                    if len(style_shape) > 2:
                        style = tf.reshape(style, [style_shape[0], tf.reduce_prod(style_shape[1:])])
                        style_shape = shape_of(style)
                    S = dense(style, units, trainable=trainable, scope='style')

                if projector == 'convolution':
                    if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                                  projector_dilations))
                    weight = get_weight(
                        [projector_kernel_size, units, material_units], projector_gain, projector_use_wscale,
                        projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)
                    weight_new = weight[None] if style is None else weight[None] * S[:, None, :, None]

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('bkdm,bkdm->bm', weight_new, weight_new) + 1e-16)
                        weight_new *= demodulator[:, None, None]

                    if fused:
                        xs = tf.reshape(tf.transpose(xs, [1, 0, 2]), [length, -1])[None]
                        weight = tf.reshape(tf.transpose(weight_new, [1, 2, 0, 3]),
                                            [projector_kernel_size, units, B * material_units])
                        X = tf.nn.conv2d(xs[:, None], weight[None],
                                         padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                         data_format='NHWC')
                        X = tf.transpose(tf.reshape(X[0], [length, B, H, material_units // H]), [1, 0, 2, 3])
                    elif style is None:
                        X = tf.nn.conv2d(xs[:, None], weight_new,
                                         padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                         data_format='NHWC')
                        X = tf.reshape(X[:, 0], [B, length, H, material_units // H])
                    else:
                        xs = tf.transpose(xs, [1, 2, 0])[None]
                        weight = tf.transpose(weight_new, [1, 2, 0, 3])
                        X = tf.nn.depthwise_conv2d(xs, weight, strides=[1, 1, 1, 1],
                                                   padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                                   data_format='NHWC')
                        X = tf.transpose(tf.reshape(X, [length, B, H, material_units // H]), [1, 0, 2, 3])

                elif projector == 'einsum_dense':
                    weight = get_weight(
                        [units, H, material_units // H], projector_gain, projector_use_wscale, projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)

                    weight_new = weight if style is None else weight[None] * S[Ellipsis, None, None]

                    if demodulate:
                        if style is None:
                            demodulator = tf.math.rsqrt(tf.einsum('dhm,dhm->hm', weight_new, weight_new) + 1e-16)
                            weight_new *= demodulator[None]
                        else:
                            demodulator = tf.math.rsqrt(tf.einsum('bdhm,bdhm->bhm', weight_new, weight_new) + 1e-16)
                            weight_new *= demodulator[:, None]

                    if fused:
                        xs = tf.transpose(xs, [1, 2, 0])
                        weight = tf.transpose(weight_new, [1, 0, 2, 3])
                        X = tf.transpose(tf.einsum('lcb,cbhd->lbhd', xs, weight), [1, 0, 2, 3])
                    elif style is None:
                        X = tf.einsum('blc,chd->blhd', xs, weight_new)
                    else:
                        X = tf.einsum('blc,bchd->blhd', xs, weight_new)

                else:
                    raise NotImplementedError

                if projector_use_bias:
                    X += tf.get_variable('bias', [material_units // H], dtype=dtype, initializer=tf.zeros_initializer(),
                                         trainable=trainable)[None, None, None]
                if projector_activation: X = projector_activation(X)
            return X

        # material projection
        Q = modulated_projection(queries, style, D_q, D_score, L_out, projector_mask_query, name='Q')
        K = modulated_projection(keys, style, D_k, D_score, L, projector_mask_key, name='K')
        V = modulated_projection(values, style, D_v, D_out, L, projector_mask_value, name='V')

        # cache
        if cache:
            K, V = [tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)]
            cache_shape = shape_of(cache['K'])
            L = cache_shape[-2] + L
            mask_key = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_key], axis=-2)
            mask_value = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_value], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'

        half_size = kernel_size // 2
        gather_size = half_size + 1 if input_relation == 'causal' else kernel_size

        indices = ((L - 1) * tf.ones([1], dtype) if cache else tf.range(L))[:, None] + (tf.range(gather_size)[None])

        K = tf.gather(tf.pad(K, [[0, 0], [half_size, half_size], [0, 0], [0, 0]]), indices, axis=1)
        K += tf.get_variable("pos_enc_K", [1, 1, gather_size, 1, D_score // heads], trainable=trainable)
        K = K * mask_key[Ellipsis, None, None, None]

        V = tf.gather(tf.pad(V, [[0, 0], [half_size, half_size], [0, 0], [0, 0]]), indices, axis=1)
        V += tf.get_variable("pos_enc_V", [1, 1, gather_size, 1, D_out // heads], trainable=trainable)
        V = V * mask_value[Ellipsis, None, None, None]

        score = tf.einsum('blhd,blkhd->blkh', Q * mask_query[Ellipsis, None, None], K) * tf.math.rsqrt(
            tf.to_float(D_score))
        score_size = tf.tile(indices[None, Ellipsis, None], [B, 1, 1, H])
        key_len = tf.to_int32(tf.einsum('bl->b', mask_key))
        attention_mask = tf.logical_and(score_size >= half_size, score_size < half_size + key_len[:, None, None, None])

        score = tf.where(attention_mask, score, tf.ones_like(score) * (-2 ** 32 + 1))
        #score = tf.nn.softmax(score, axis=-2)
        score = score_activation(score)

        out_ = tf.einsum('blkh,blkhd->blhd', score, V)
        out_ = out_ - (tf.einsum('blhd->bhd', out_) / tf.einsum('bl->b', mask_query)[:, None, None])[:, None]
        out_ = out_ * mask_query[Ellipsis, None, None]

        '''
        def restore_projection(xs, units, restore_units, length, mask, name='restore'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size, projector_dilations))
                X = convolution(tf.reshape(xs, [B, length, units]), 1, restore_units, projector_kernel_size, projector_strides, projector_dilations, padding=('valid' if input_relation=='causal' else 'same'),
                    mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_initializer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name)
            elif projector == 'dense':
                X = dense(tf.reshape(xs, [B, length, units]), restore_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name)
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_'+name):
                    kernel = tf.get_variable('kernel', shape=[H, units//H, restore_units], dtype=dtype,
                            initializer=projector_weight_initializer, regularizer=projector_weight_regularizer, trainable=trainable, constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: kernel = projector_weight_normalizer(kernel)
                    X = tf.einsum("blho,hor->blr", xs, kernel)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[restore_units], dtype=dtype,
                                initializer=projector_bias_initializer, regularizer=projector_bias_regularizer, trainable=trainable, constraint=projector_bias_constraint)
                        X = X + bias[None, None]
            else: raise NotImplementedError
            return X
        '''

        def modulated_restoration(xs, units, restore_units, mask, name='O'):

            with tf.variable_scope(name):
                if projector == 'convolution':
                    if input_relation == 'causal':
                        xs = tf.pad(xs, causal_padding(xs, projector_kernel_size, projector_dilations))
                    else:
                        same_pad = projector_kernel_size // 2
                        xs = tf.pad(xs, [[0, 0], [same_pad, same_pad], [0, 0], [0, 0]])

                    weight = get_weight(
                        [projector_kernel_size, H, units, restore_units], projector_gain, projector_use_wscale,
                        projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('khmo,khmo->o', weight, weight) + 1e-16)
                        weight *= demodulator[None, None, None]

                    X = tf.nn.conv2d(xs, weight, padding='VALID', data_format='NHWC')
                    X = tf.squeeze(X, axis=2)

                elif projector == 'einsum_dense':
                    weight = get_weight(
                        [H, units, restore_units], projector_gain, projector_use_wscale, projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('hmo,hmo->o', weight, weight) + 1e-16)
                        weight *= demodulator[None, None]

                    X = tf.einsum('blhm,hmo->blo', xs, weight)

                else:
                    raise NotImplementedError

                if projector_use_bias:
                    X += tf.get_variable('bias', [restore_units], dtype=dtype, initializer=tf.zeros_initializer(),
                                         trainable=trainable)[None, None]
                if mask is not None: X *= mask[Ellipsis, None]
            return X

        out_ = modulated_restoration(out_, D_out // H, D_out, mask_query, name='restore')
        out_ = tf.reshape(out_, Bs_q + [L_out, D_out])
        if activation: out_ = activation(out_)

    # return out_, cache, score
    return out_


def random_feature_attention(queries,
                             keys,
                             values,
                             input_relation='self',
                             heads=4,
                             units=None,
                             map_units=None,
                             feature_map='arc-cosine',
                             outer_mask=None,
                             inner_mask=None,
                             final_mask=None,
                             cache=None,
                             n_batch_dims=1,
                             use_recency_bias=False,
                             activation=None,
                             projector='convolution',
                             projector_kernel_size=3,
                             projector_strides=1,
                             projector_dilations=None,
                             projector_mask_query=None,
                             projector_mask_key=None,
                             projector_mask_value=None,
                             projector_groups=1,
                             projector_use_bias=True,
                             projector_activation=None,
                             projector_weight_initializer=tf.glorot_uniform_initializer(),
                             projector_weight_regularizer=None,
                             projector_weight_constraint=None,
                             projector_weight_normalizer=None,
                             projector_bias_initializer=tf.zeros_initializer(),
                             projector_bias_regularizer=None,
                             projector_bias_constraint=None,
                             trainable=True,
                             scope='random_feature_attention'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))
    batch_dims = [axis for axis in range(n_batch_dims)]

    outer_shape = shape_of(queries)
    if not isinstance(units, int): units = outer_shape[-1]
    if units % heads != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(heads, units))

    inner_shape = shape_of(keys)
    if not isinstance(map_units, int): map_units = inner_shape[-1]
    if map_units % heads != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            heads, map_units))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    if not feature_map in ['gaussian', 'arc-cosine']: raise NotImplementedError(
        "given method {} of shift-invariant kernels is unsupported.".format(feature_map))

    with tf.variable_scope(scope):
        # graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype

        # with tf.control_dependencies([tf_print(f'{graph_name}/query', queries), tf_print(f'{graph_name}/key', keys)]):
        #    queries = tf.identity(queries)

        # projection
        if projector == 'convolution':
            if input_relation == 'causal': queries, keys, values = [
                tf.pad(in_, causal_padding(in_, projector_kernel_size, projector_dilations)) for in_ in
                [queries, keys, values]]
            padding = 'valid' if input_relation == 'causal' else 'same'
            Q, K, V = [tf.transpose(
                tf.reshape(convolution(in_, 1, numerics, projector_kernel_size, projector_strides, projector_dilations,
                                       padding=padding, in_mask=in_mask, out_mask=out_mask, groups=projector_groups,
                                       use_bias=projector_use_bias, activation=projector_activation,
                                       weight_initializer=projector_weight_initializer,
                                       weight_regularizer=projector_weight_regularizer,
                                       weight_constraint=projector_weight_constraint,
                                       weight_normalizer=projector_weight_normalizer,
                                       bias_initializer=projector_bias_initializer,
                                       bias_regularizer=projector_bias_regularizer,
                                       bias_constraint=projector_bias_constraint, trainable=trainable,
                                       scope='projection_' + name),
                           shape[:-1] + [heads, numerics // heads]),
                batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])
                for in_, name, numerics, shape, in_mask, out_mask in
                [(queries, 'Q', map_units, outer_shape, projector_mask_query, outer_mask),
                 (keys, 'K', map_units, inner_shape, projector_mask_key, inner_mask),
                 (values, 'V', units, inner_shape, projector_mask_value,
                  tf.reduce_max(inner_mask, axis=-1, keepdims=True))]]

        elif projector == 'dense':
            Q, K, V = [tf.transpose(tf.reshape(
                dense(in_, numerics, in_mask=in_mask, out_mask=out_mask, groups=projector_groups,
                      use_bias=projector_use_bias, activation=projector_activation,
                      weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer,
                      weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                      bias_initializer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                      bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name),
                shape[:-1] + [heads, numerics // heads]),
                batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])
                for in_, name, numerics, shape, in_mask, out_mask in
                [(queries, 'Q', map_units, outer_shape, projector_mask_query, outer_mask),
                 (keys, 'K', map_units, inner_shape, projector_mask_key, inner_mask),
                 (values, 'V', units, inner_shape, projector_mask_value,
                  tf.reduce_max(inner_mask, axis=-1, keepdims=True))]]

        else:
            raise NotImplementedError
        # with tf.control_dependencies([tf_print(f'{graph_name}/Q', Q), tf_print(f'{graph_name}/K', K)]):
        #    Q = tf.identity(Q)

        if inner_mask is None: inner_mask = tf.ones(inner_shape[:-1] + [map_units], dtype=dtype)
        if outer_mask is None: outer_mask = tf.ones(outer_shape[:-1] + [map_units],
                                                    dtype=dtype)  # batch_dims + [length]
        if final_mask is None: final_mask = tf.ones(outer_shape[:-1] + [units], dtype=dtype)
        inner_mask = tf.transpose(tf.reshape(inner_mask, inner_shape[:-1] + [heads, map_units // heads]),
                                  batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])
        outer_mask = tf.transpose(tf.reshape(outer_mask, outer_shape[:-1] + [heads, map_units // heads]),
                                  batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])
        final_mask = tf.transpose(tf.reshape(final_mask, outer_shape[:-1] + [heads, units // heads]),
                                  batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])

        """
        currently:
        Q -> outer_shape[:n_batch_dims] + [heads, outer_length, map_units//heads]
        K -> inner_shape[:n_batch_dims] + [heads, inner_length, map_units//heads]
        V -> inner_shape[:n_batch_dims] + [heads, inner_length, units//heads]
        """

        # cache
        if cache:
            K, V = tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)
            inner_mask = tf.concat([tf.ones_like(cache['K'], dtype=dtype), inner_mask], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'
            inner_shape = shape_of(K)
        # with tf.control_dependencies([tf_print(f'{graph_name}/projected_Q', Q), tf_print(f'{graph_name}/projected_K', K)]): Q = tf.identity(Q)

        Q_denominator = tf.math.sqrt(tf.reduce_sum(tf.math.square(Q), axis=-1, keepdims=True))
        Q_norm_1 = Q * (tf.math.maximum(Q_denominator, 1e-12) ** -1.0) * outer_mask
        K_denominator = tf.math.sqrt(tf.reduce_sum(tf.math.square(K), axis=-1, keepdims=True))
        K_norm_1 = K * (tf.math.maximum(K_denominator, 1e-12) ** -1.0) * inner_mask
        # with tf.control_dependencies([tf_print(f'{graph_name}/Q_norm_1', Q_norm_1), tf_print(f'{graph_name}/K_norm_1', K_norm_1)]):
        #    Q_norm_1 = tf.identity(Q_norm_1)

        # main
        # feature_weight_Q = tf.get_variable(
        #        'gate_feature_Q', shape=[1]*n_batch_dims + [1,  1, map_units//heads, map_units//heads], dtype=dtype,
        #        initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), trainable=False)     # 1, 1, ..., 1, 1, U_m//H, U_m//H
        # feature_weight_K = tf.get_variable(
        #        'gate_feature_K', shape=[1]*n_batch_dims + [1,  1, map_units//heads, map_units//heads], dtype=dtype,
        #        initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), trainable=False)     # 1, 1, ..., 1, 1, U_m//H, U_m//H
        # random_map_Q = tf.squeeze(feature_weight_Q, axis=[-3])
        # random_map_K = tf.squeeze(feature_weight_K, axis=[-3])

        # real part
        # random_map_Q = tf.random.normal(outer_shape[:-2] + [heads, map_units//heads, map_units//heads], mean=0.0, stddev=1.0, dtype=dtype)
        # random_map_K = tf.random.normal(inner_shape[:-2] + [heads, map_units//heads, map_units//heads], mean=0.0, stddev=1.0, dtype=dtype)
        random_weight = tf.random.normal(outer_shape[:-2] + [heads, map_units // heads, map_units // heads], mean=0.0,
                                         stddev=1.0, dtype=dtype)
        learned_sigma = tf.get_variable(
            'sigma', shape=[1] * n_batch_dims + [1, map_units // heads, 1], dtype=dtype,
            initializer=tf.constant_initializer(value=(map_units // heads) ** 0.25), trainable=trainable)

        scaler = tf.math.rsqrt(tf.cast(map_units // heads, dtype))
        Q_gated = tf.linalg.matmul(Q_norm_1, learned_sigma * random_weight)
        K_gated = tf.linalg.matmul(K_norm_1, learned_sigma * random_weight)
        # with tf.control_dependencies([tf_print(f'{graph_name}/Q*weight', Q_gated), tf_print(f'{graph_name}/K*weight', K_gated)]):
        #    Q_gated = tf.identity(Q_gated)

        if feature_map == 'gaussian':
            # Q_gated = 2 * np.pi * Q_gated
            # K_gated = 2 * np.pi * K_gated
            Q_gated = scaler * tf.concat([tf.math.sin(Q_gated), tf.math.cos(Q_gated)], axis=-1)
            K_gated = scaler * tf.concat([tf.math.sin(K_gated), tf.math.cos(K_gated)], axis=-1)
            # with tf.control_dependencies([tf_print('Q_gated', Q_gated), tf_print('K_gated', K_gated)]):
            #    Q_gated = tf.identity(Q_gated)

        elif feature_map == 'arc-cosine':
            Q_gated = scaler * relu(Q_gated)
            K_gated = scaler * relu(K_gated)
        # with tf.control_dependencies([tf_print(f'{graph_name}/unit_Q', Q_as_unit_vector), tf_print(f'{graph_name}/unit_K', K_as_unit_vector)]): Q_as_unit_vector = tf.identity(Q_as_unit_vector)

        # with tf.control_dependencies([tf_print(f'{graph_name}/Q_gated', Q_gated), tf_print(f'{graph_name}/K_gated', K_gated)]):
        #    Q_gated = tf.identity(Q_gated)

        """
        NOTICE:
        if memory more capable - use currently code, which composed of elementwise multiplication. (Faster, but intermediate tensor has larger shape)
        if memory not capable - change elementwise multiplication terms into matrix multiplication. (Slower, but intermediate tensor has smaller shape)
        """

        # when recency bias needed (positional encodings)
        if use_recency_bias:
            gate_material = tf.transpose(tf.reshape(keys, inner_shape[:-1] + [heads, inner_shape[-1] // heads]),
                                         batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2])
            scalar_gates = dense(K, 1, in_mask=inner_mask, out_mask=tf.reduce_max(inner_mask, axis=-1, keepdims=True),
                                 use_bias=True, activation=sigmoid, trainable=trainable,
                                 scope='scalar_gate')  # B0, B1, ... , Bn, H, intermediate_length, 1
            with tf.control_dependencies([tf_print('sc_gate', scalar_gates)]):
                scalar_gates = tf.identity(scalar_gates)
            anti_gates = 1.0 - scalar_gates

            if feature_map == 'gaussian':
                inner_mask = tf.tile(inner_mask, [1] * (n_batch_dims + 2) + [2])
                outer_mask = tf.tile(outer_mask, [1] * (n_batch_dims + 2) + [2])

            G = tf.math.cumprod(scalar_gates, axis=-2)
            ln_G = tf.math.maximum(tf.math.log(G), (-2.0 ** 31 + 1.0))

            if input_relation == 'causal':
                anti_K = anti_gates * K_gated
                sign_anti_K = tf.math.sign(anti_K)
                ln_abs_anti_K = tf.math.maximum(tf.math.log(sign_anti_K * anti_K), (-2.0 ** 31 + 1.0))
                divided_K = tf.math.minimum(tf.math.exp(ln_abs_anti_K - ln_G), 2.0 ** 31 - 1.0) * sign_anti_K
                z = G * tf.math.cumsum(divided_K, axis=-2)
                score = tf.reduce_sum(Q_gated * z, axis=-1, keepdims=True)

                K_cross_V = tf.expand_dims(K_gated, axis=-1) * tf.expand_dims(V, axis=-2)
                anti_cross = tf.expand_dims(anti_gates, axis=-1) * K_cross_V
                sign_anti_cross = tf.math.sign(anti_cross) * tf.expand_dims(inner_mask, axis=-1)
                ln_abs_anti_cross = tf.math.maximum(tf.math.log(sign_anti_cross * anti_cross), (-2.0 ** 31 + 1.0))
                divided_cross = tf.math.minimum(
                    tf.math.exp(ln_abs_anti_cross - tf.expand_dims(ln_G * inner_mask, axis=-1)),
                    2.0 ** 31 - 1.0) * sign_anti_cross
                match_table = tf.expand_dims(G, axis=-1) * tf.math.cumsum(divided_cross, axis=-3)
                out_ = tf.math.divide_no_nan(
                    tf.squeeze(tf.linalg.matmul(tf.expand_dims(Q_gated * outer_mask, axis=-2), match_table), axis=[-2]),
                    score)

                out_ = tf.where(tf.equal(final_mask, 0.0), tf.zeros_like(out_, dtype=dtype), out_)

                """
                with tf.control_dependencies([tf_print(f'{graph_name}/Q_masked', Q_masked), tf_print(f'{graph_name}/K_masked', K_masked)]):
                    with tf.control_dependencies([tf_print(f'{graph_name}/scalar_gates', scalar_gates), tf_print(f'{graph_name}/anti_gates', anti_gates)]):
                        with tf.control_dependencies([tf_print(f'{graph_name}/division_mask', division_mask)]):
                            with tf.control_dependencies([tf_print(f'{graph_name}/G', G),
                                                          #tf_print(f'{graph_name}/max_pad_G', max_padded_G),
                                                          tf_print(f'{graph_name}/ln_G', ln_G)]):
                                with tf.control_dependencies([tf_print(f'{graph_name}/anti_gate*K_masked', anti_K), tf_print(f'{graph_name}/sign_anti_K', sign_anti_K)]):
                                    with tf.control_dependencies([tf_print(f'{graph_name}/div_K', divided_K)]):
                                        with tf.control_dependencies([tf_print(f'{graph_name}/z', z)]):
                                            with tf.control_dependencies([tf_print(f'{graph_name}/score', score)]):
                                                with tf.control_dependencies([tf_print(f'{graph_name}/K_cross_V', K_cross_V)]):
                                                    with tf.control_dependencies([tf_print(f'{graph_name}/anti_gate*K_cross_V', anti_cross), tf_print(f'{graph_name}/sign_anti_cross', sign_anti_cross)]):
                                                        with tf.control_dependencies([tf_print(f'{graph_name}/div_cross', divided_cross)]):
                                                            with tf.control_dependencies([tf_print(f'{graph_name}/match_table', match_table)]):
                                                                with tf.control_dependencies([tf_print(f'{graph_name}/unshaped_out', out_)]):
                                                                    out_ = tf.identity(out_)
                """

            elif (input_relation == 'self') or (input_relation == 'cross'):
                intermediate_shape = shape_of(K)

                G_last = tf.slice(G, [0] * (n_batch_dims + 1) + [intermediate_shape[n_batch_dims + 1] - 1, 0],
                                  intermediate_shape[:n_batch_dims + 1] + [1, 1])
                anti_K = anti_gates * K_gated
                sign_anti_K = tf.math.sign(anti_K) * inner_mask
                ln_abs_anti_K = tf.math.maximum(tf.math.log(sign_anti_K * anti_K), (-2.0 ** 31 + 1.0))
                divided_K = tf.math.minimum(tf.math.exp(ln_abs_anti_K - ln_G), 2.0 ** 31 - 1.0) * sign_anti_K
                z = G_last * tf.math.reduce_sum(divided_K, axis=-2, keepdims=True)
                score = tf.reduce_sum(Q_gated * z, axis=-1, keepdims=True)

                K_cross_V = tf.expand_dims(K_gated, axis=-1) * tf.expand_dims(V, axis=-2)
                anti_cross = tf.expand_dims(anti_gates, axis=-1) * K_cross_V
                sign_anti_cross = tf.math.sign(anti_cross) * tf.expand_dims(inner_mask, axis=-1)
                ln_abs_anti_cross = tf.math.maximum(tf.math.log(sign_anti_cross * anti_cross), (-2.0 ** 31 + 1.0))
                divided_cross = tf.math.minimum(
                    tf.math.exp(ln_abs_anti_cross - tf.expand_dims(ln_G * inner_mask, axis=-1)),
                    2.0 ** 31 - 1.0) * sign_anti_cross
                match_table = G_last * tf.math.reduce_sum(divided_cross, axis=-3)
                out_ = tf.math.divide_no_nan(tf.linalg.matmul(Q_gated * outer_mask, match_table), score)

                out_ = tf.where(tf.equal(final_mask, 0.0), tf.zeros_like(out_, dtype=dtype), out_)

                """
                with tf.control_dependencies([tf_print(f'{graph_name}/Q_masked', Q_masked), tf_print(f'{graph_name}/K_masked', K_masked)]):
                    with tf.control_dependencies([tf_print(f'{graph_name}/scalar_gates', scalar_gates), tf_print(f'{graph_name}/anti_gates', anti_gates)]):
                        with tf.control_dependencies([tf_print(f'{graph_name}/division_mask', division_mask)]):
                            with tf.control_dependencies([tf_print(f'{graph_name}/G', G),
                                                          #tf_print(f'{graph_name}/max_pad_G', max_padded_G),
                                                          tf_print(f'{graph_name}/ln_G', ln_G), tf_print(f'{graph_name}/G_last', G_last)]):
                                with tf.control_dependencies([tf_print(f'{graph_name}/anti_gate*K_masked', anti_K), tf_print(f'{graph_name}/sign_anti_K', sign_anti_K)]):
                                    with tf.control_dependencies([tf_print(f'{graph_name}/div_K', divided_K)]):
                                        with tf.control_dependencies([tf_print(f'{graph_name}/z', z)]):
                                            with tf.control_dependencies([tf_print(f'{graph_name}/score', score)]):
                                                with tf.control_dependencies([tf_print(f'{graph_name}/K_cross_V', K_cross_V)]):
                                                    with tf.control_dependencies([tf_print(f'{graph_name}/anti_gate*K_cross_V', anti_cross), tf_print(f'{graph_name}/sign_anti_cross', sign_anti_cross)]):
                                                        with tf.control_dependencies([tf_print(f'{graph_name}/div_cross', divided_cross)]):
                                                            with tf.control_dependencies([tf_print(f'{graph_name}/match_table', match_table)]):
                                                                with tf.control_dependencies([tf_print(f'{graph_name}/unshaped_out', out_)]):
                                                                    out_ = tf.identity(out_)
                """

            else:
                raise NotImplementedError

            """
            # logsumexp version - can't use since K or K cross V can have negetive values
            ln_anti_gates = tf.math.log(1. - scalar_gates)     # shape[:n_batch_dims] + [heads, intermediate_length, 1]

            if input_relation == 'causal':
                ln_G = tf.math.cumsum(tf.math.log(scalar_gates), axis=-2)     # shape[: n_batch_dims] + [heads, intermediate_length, 1]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_G', ln_G)]):
                    ln_G = tf.identity(ln_G)
                ln_z = ln_G + tf.math.cumulative_logsumexp(tf.math.log(K) + ln_anti_gates - ln_G, axis=-2)     # shape[:n_batch_dims] + [heads, intermediate_length, 2*map_units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_z', ln_z)]):
                    ln_z = tf.identity(ln_z)
                score = tf.reduce_sum(Q_masked * tf.math.exp(ln_z), axis=-1, keepdims=True)     # shape[:n_batch_dims] + [heads, intermediate_length, 1]
                with tf.control_dependencies([tf_print(f'{graph_name}/score', score)]):
                    score = tf.identity(score)
                k_cross_v = tf.expand_dims(K, axis=-1) * tf.expand_dims(V, axis=-2)     # shape[:n_batch_dims] + [heads, intermediate_length, 2*map_units//heads, units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/k_cross_v', k_cross_v)]):
                    k_cross_v = tf.identity(k_cross_v)
                ln_match_table = tf.expand_dims(ln_G, axis=-1) + tf.math.cumulative_logsumexp(tf.math.log(k_cross_v) + tf.expand_dims(ln_anti_gates - ln_G, axis=-1), axis=-3)
                    # shape[:n_batch_dims] + [heads, intermediate_length, 2*map_units//heads, units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_match_table', ln_match_table)]):
                    ln_match_table = tf.identity(ln_match_table)
                out_ = tf.math.divide_no_nan(tf.squeeze(tf.linalg.matmul(tf.expand_dims(Q_masked, axis=-2), tf.math.exp(ln_match_table)), axis=[-2]), score, name='safe_division')
                with tf.control_dependencies([tf_print(f'{graph_name}/unshaped_out', out_)]):
                    out_ = tf.identity(out_)
            elif (input_relation == 'self') or (input_relation == 'cross'):
                ln_G = tf.math.cumsum(tf.math.log(scalar_gates), axis=-2)     # shape[:n_batch_dims] + [heads, intermediate_length, 1]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_G', ln_G)]):
                    ln_G = tf.identity(ln_G)
                ln_G_last = tf.slice(ln_G, [0]*(n_batch_dims + 1) + [intermediate_shape[n_batch_dims+1]-1, 0], intermediate_shape[:n_batch_dims+1] + [1, 1])     # shape[:n_batch_dims] + [heads, 1, 1]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_G_last', ln_G_last)]):
                    ln_G_last = tf.identity(ln_G_last)
                ln_z = ln_G_last + tf.math.reduce_logsumexp(tf.math.log(K) + ln_anti_gates - ln_G, axis=-2, keepdims=True)     # B0, B1, ... , Bn, H, 1, 2*map_units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_z', ln_z)]):
                    ln_z = tf.identity(ln_z)
                score = tf.linalg.matmul(Q_masked, ln_z, transpose_b=True)     # shape[:n_batch_dims] + [heads, outer_length, 1]
                with tf.control_dependencies([tf_print(f'{graph_name}/score', score)]):
                    score = tf.identity(score)
                k_cross_v = tf.expand_dims(K, axis=-1) * tf.expand_dims(V, axis=-2)     # shape[:n_batch_dims] + [heads, inner_length, 2*map_units//heads, units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/k_cross_v', k_cross_v)]):
                    k_cross_v = tf.identity(k_cross_v)
                ln_match_table = ln_G_last + tf.math.reduce_logsumexp(tf.math.log(k_cross_v) + tf.expand_dims(ln_anti_gates - ln_G, axis=-1), axis=-3)
                    # shape[:n_batch_dims] + [heads, 2*map_units//heads, units//heads]
                with tf.control_dependencies([tf_print(f'{graph_name}/ln_match_table', ln_match_table)]):
                    ln_match_table = tf.identity(ln_match_table)
                out_ = tf.math.divide_no_nan(tf.linalg.matmul(Q_masked, tf.math.exp(ln_match_table)), score, name='safe_division')
                with tf.control_dependencies([tf_print(f'{graph_name}/unshaped_out', out_)]):
                    out_ = tf.identity(out_)
            else: raise NotImplementedError
            """
            """
            # direct vertsion : fully generalized terms, but memory runs out
            scalar_gates = dense(K, 1, activation=sigmoid, use_bias=True, trainable=trainable, scope='scalar_gate'))     # shape[:n_batch_dims] + [heads, inner_length, 1]
            former_range = tf.tile(tf.reshape(tf.range(intermediate_shape[-2])[:, None], [1]*n_batch_dims + [1, intermediate_shape[-2], 1]), intermediate_shape[:n_batch_dims] + [heads, 1, 1])\
                         * tf.reshape(inner_mask, shape_of(inner_mask)[:-1] + [1, intermediate_shape[-2], 1])
            latter_range = tf.tile(tf.reshape(tf.range(intermediate_shape[-2])[None], [1]*n_batch_dims + [1, 1, intermediate_shape[-2]]), intermediate_shape[:n_batch_dims] + [heads, 1, 1])\
                         * tf.reshape(inner_mask, shape_of(inner_mask)[:-1] + [1, 1, intermediate_shape[-2]])
            indices_placement = tf.tile(tf.expand_dims(former_range > latter_range, axis=-1), intermediate_shape[:n_batch_dims] + [1, 1, 1, 2*map_units//heads])
            indices_product = tf.tile(tf.expand_dims(former_range < latter_range, axis=-1), intermediate_shape[:n_batch_dims] + [1, 1, 1, 2*map_units//heads])

            tiled_gates = tf.tile(tf.expand_dims(scalar_gates, axis=-2), [1]*n_batch_dims + [1, 1, intermediate_shape[-2], 1])
                # intermediate_shape[:n_batch_dims] + [heads, inner_length, inner_lenngth, 2*map_units//heads]
            mask = tf.where(indices_placement, tiled_gates, tf.ones_like(tiled_gates))
            mask = tf.where(indices_product, tf.zeros_like(mask), tf.math.cumprod(mask, axis=-3))
                # mask : batch_dims + [heads, inner_length, inner_length, 2*map_units//heads]

            if input_relation == 'causal':
                z = tf.math.reduce_sum(mask * tf.expand_dims((1 - scalar_gates) * K, axis=-3), axis=-2)
                    # shape[:n_batch_dims] + [heads, length, 2*map_units//heads]
                match_table = tf.math.reduce_sum(mask * tf.expand_dims(tf.expand_dims(K, axis=-1) * tf.expand_dims(V, axis=-2), axis=-4), axis=-3)
                    # shape[:n_batch_dims] + [heads, length, 2*map_units//heads, units//heads]
                score = tf.linalg.matmul(Q_masked * z, axis=-1, keepdims=True)
                    # shape[:n_batch_dims] + [heads, length, 1]
                out = tf.math.divide_no_nan(tf.reduce_sum(tf.expand_dims(Q_masked, axis=-1) * match_table, axis=-2), score, name='safe_division')
            elif (input_relation == 'self') or (input_relation == 'cross'):
                z = tf.math.reduce_sum(mask * tf.expand_dims((1 - scalar_gates) * K, axis=-3), axis=[-3, -2])
                    # intermediate_shape[:n_batch_dims] + [heads, 2*map_units//heads]
                match_table = tf.math.reduce_sum(mask * tf.expand_dims(tf.expand_dims(K, axis=-1) * tf.expand_dims(V, axis=-2), axis=-4), axis=[-4, -3])
                    # intermediate_shape[:n_batch_dims] + [heads, 2*map_units//heads, units//heads]
                score = tf.math.reduce_sum(Q_masked * tf.expand_dims(z, axis=-2), axis=-1, keepdims=True)
                    # shape[:n_batch_dims] + [heads, length, 1]
                out_ = tf.math.divide_no_nan(tf.linalg.matmul(Q_masked, match_table), score, name='safe_division')
            else: raise NotImplementedError
            """

        else:
            if feature_map == 'gaussian':
                inner_mask = tf.tile(inner_mask, [1] * (n_batch_dims + 2) + [2])
                outer_mask = tf.tile(outer_mask, [1] * (n_batch_dims + 2) + [2])

            if input_relation == 'causal':
                z = tf.math.cumsum(K_gated,
                                   axis=-2)  # shape[:n_batch_dims] + [heads, intermediate_length, 2*map_units//heads]
                score = tf.reduce_sum(Q_gated * z, axis=-1,
                                      keepdims=True)  # intermediate_shape==outer_shape | B0, ..., Bn, H, length, 1

                match_table = tf.math.cumsum(tf.expand_dims(K_gated * inner_mask, axis=-1) * tf.expand_dims(V, axis=-2),
                                             axis=-3)  # B0, ..., Bn, H, intermediate_length, 2*map_units//heads, units//heads
                out_ = tf.squeeze(tf.linalg.matmul(tf.expand_dims(Q_gated * outer_mask, axis=-2), match_table),
                                  axis=[-2]) * (tf.math.maximum(score, 1e-12) ** -1.0)

                out_ = tf.where(tf.equal(final_mask, 0.0), tf.zeros_like(out_, dtype=dtype), out_)

            elif (input_relation == 'self') or (input_relation == 'cross'):
                z = tf.math.reduce_sum(K_gated, axis=-2,
                                       keepdims=True)  # intermediate_shape[:n_batch_dims] + [heads, 1, 2*map_units//heads]
                # with tf.control_dependencies([tf_print(f'{graph_name}/z', z)]): z = tf.identity(z)
                score = tf.math.reduce_sum(Q_gated * z, axis=-1,
                                           keepdims=True)  # outer_shape[:n_batch_dims] + [heads, outer_length, 1]
                # with tf.control_dependencies([tf_print(f'{graph_name}/score', score)]): score = tf.identity(score)

                match_table = tf.linalg.matmul(K_gated * inner_mask, V,
                                               transpose_a=True)  # intermediate_shape[:n_batch_dims] + [heads, 2*map_units//heads, units//heads]
                # with tf.control_dependencies([tf_print(f'{graph_name}/match_table', match_table)]): match_table = tf.identity(match_table)
                out_ = tf.linalg.matmul(Q_gated * outer_mask, match_table) * (tf.math.maximum(score, 1e-12) ** -1.0)
                # with tf.control_dependencies([tf_print(f'{graph_name}/out_', out_), tf_print(f'{graph_name}/rev_score', (tf.maximum(score, 1e-12) ** -1.0))]): out_ = tf.identity(out_)

                out_ = tf.where(tf.equal(final_mask, 0.0), tf.zeros_like(out_, dtype=dtype), out_)
                # with tf.control_dependencies([tf_print(f'{graph_name}/masked_out', out_)]): out_ = tf.identity(out_)

            else:
                raise NotImplementedError

        out_ = tf.reshape(tf.transpose(out_, batch_dims + [n_batch_dims + 1, n_batch_dims, n_batch_dims + 2]),
                          outer_shape[:-1] + [units])
        if activation: out_ = activation(out_)
        # with tf.control_dependencies([tf_print(f'{graph_name}/rfa_out', out_)]): out_ = tf.identity(out_)

    return out_, cache, score


def favorplus(queries,
              keys,
              values,
              units=None,
              map_units=None,
              heads=4,
              input_relation='self',
              feature_map='relu',
              feature_projection=True,
              mask_query=None,
              mask_key=None,
              mask_value=None,
              cache=None,
              use_recency_bias=False,
              activation=None,
              seed_value=None,
              projector='einsum_dense',
              projector_kernel_size=3,
              projector_strides=1,
              projector_dilations=None,
              projector_mask_query=None,
              projector_mask_key=None,
              projector_mask_value=None,
              projector_groups=1,
              projector_use_bias=True,
              projector_activation=None,
              projector_weight_initializer=tf.glorot_uniform_initializer(),
              projector_weight_regularizer=None,
              projector_weight_constraint=None,
              projector_weight_normalizer=None,
              projector_bias_initializer=tf.zeros_initializer(),
              projector_bias_regularizer=None,
              projector_bias_constraint=None,
              trainable=True,
              scope='favorplus'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))

    *Bs_q, L_out, D_q = shape_of(queries)
    *Bs, L, D_k = shape_of(keys)
    *Bs_v, L_v, D_v = shape_of(values)

    # assert Bs_q == Bs and Bs_v == Bs, f"Batch of queries : {Bs_q}  Batch of keys : {Bs}  Batch of values : {Bs_v}"
    # assert L_v == L

    # n_B_axis = len(Bs)
    # B_axis = [axis for axis in range(n_B_axis)]

    D_qk = max(D_q, D_k)

    H = heads if isinstance(heads, int) else 1

    D_out = units if isinstance(units, int) else D_q
    if D_out % H != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(H, D_out))

    D_map = map_units if isinstance(map_units, int) else D_qk // 2
    if D_map % H != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            H, D_map))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    if not feature_map in ['relu', 'softmax']: raise NotImplementedError(
        "given method {} of shift-invariant kernels is unsupported.".format(feature_map))

    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype
        B = tf.reduce_prod(Bs, keepdims=False)

        # batch shape change
        queries, keys, values = [tf.reshape(in_, [B, length, numerics]) for in_, length, numerics in
                                 [(queries, L_out, D_q), (keys, L, D_k), (values, L, D_v)]]

        mask_query = tf.ones([B, L_out], dtype=dtype) if mask_query is None else tf.reshape(mask_query, [B, L_out])
        mask_key = tf.ones([B, L], dtype=dtype) if mask_key is None else tf.reshape(mask_key, [B, L])
        mask_value = tf.ones([B, L], dtype=dtype) if mask_value is None else tf.reshape(mask_value, [B, L])

        if projector_mask_query is not None: projector_mask_query = tf.reshape(projector_mask_query, [B, L_out])
        if projector_mask_key is not None: projector_mask_key = tf.reshape(projector_mask_key, [B, L])
        if projector_mask_value is not None: projector_mask_value = tf.reshape(projector_mask_value, [B, L])

        def material_projection(xs, units, material_units, length, mask, name='X'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = tf.reshape(
                    convolution(xs, 1, material_units, projector_kernel_size, projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'dense':
                X = tf.reshape(
                    dense(xs, material_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias,
                          activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name),
                    [B, length, H, material_units // H])
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    kernel = tf.get_variable('kernel', shape=[units, H, material_units // H], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer: kernel = projector_weight_normalizer(kernel)
                    X = tf.einsum("blc,chd->blhd", xs, kernel)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[material_units // H], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None, None]
                    if projector_activation: X = projector_activation(X)
            else:
                X = tf.reshape(xs, [B, length, material_units])
            return X

        # material projection
        Q = material_projection(queries, D_q, D_qk, L_out, projector_mask_query, 'Q')
        K = material_projection(keys, D_k, D_qk, L, projector_mask_key, 'K')
        V = material_projection(values, D_v, D_out, L, projector_mask_value, 'V')

        # projection
        def _create_products_of_givens_rotations(d, seed=None):
            n_givens_rotations = d * int(math.ceil(math.log(float(d))))
            q = np.eye(d, d)
            if seed is not None: np.random.seed(seed)
            for _ in range(n_givens_rotations):
                random_angle = math.pi * np.random.uniform()
                random_indices = np.random.choice(d, 2)
                index_i = min(random_indices[0], random_indices[1])
                index_j = max(random_indices[0], random_indices[1])
                slice_i = q[index_i]
                slice_j = q[index_j]
                new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
                new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
                q[index_i] = new_slice_i
                q[index_j] = new_slice_j
            return tf.cast(tf.constant(q), dtype=dtype)

        def _create_projection_matrix(m, d, seed, scaling=0, struct_mode=False):
            n_full_blocks = m // d
            block_list = []
            current_seed = seed
            for _ in range(n_full_blocks):
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q)
                current_seed += 1
            remaining_rows = m - n_full_blocks * d
            if remaining_rows > 0:
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q[0:remaining_rows])
            final_matrix = tf.concat(block_list, axis=0)
            current_seed += 1

            if scaling == 0:
                multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
            elif scaling == 1:
                multiplier = tf.math.sqrt(float(d)) * tf.ones(float(m))
            else:
                raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)
            return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)

        if feature_projection:
            # seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(Q) * 1e8))
            # seed = tf.dtypes.cast(seed, tf.int32)
            # projection_matrix = _create_projection_matrix(seed=seed)
            seed = random.randint(0, (2 ** 31) - 1) if seed_value is None else seed_value
            projection_matrix = _create_projection_matrix(D_map, D_qk // H, seed=seed)
        else:
            projection_matrix = None

        if use_recency_bias:
            gate = dense(keys, H, mask=projector_mask_key, groups=projector_groups, use_bias=True, activation=sigmoid,
                         trainable=trainable, scope='gate_dense')
        else:
            gate = None

        # cache
        if cache:
            K, V = [tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)]
            if use_recency_bias:
                gate = tf.concat([cache['gate'], gate], -2)
                cache['gate'] = gate
            cache_shape = shape_of(cache['K'])
            L = cache_shape[-2] + L
            mask_key = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_key], axis=-2)
            mask_value = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_value], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'

        def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):

            del is_query
            if projection_matrix is None:
                return relu(data) + numerical_stabilizer
            else:
                ratio = 1.0 * tf.math.rsqrt(tf.cast(D_map, tf.float32))
                data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
                return relu(data_dash) + numerical_stabilizer

        def softmax_kernel_transformation(data, units_data, is_query, projection_matrix=None,
                                          numerical_stabilizer=0.000001):

            data_normalizer = 1.0 * tf.math.rsqrt(tf.math.sqrt(tf.cast(units_data, tf.float32)))
            data = data_normalizer * data

            ratio = 1.0 * tf.math.rsqrt(tf.cast(D_map, tf.float32))
            data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
            diag_data = tf.math.square(data)
            diag_data = tf.math.reduce_sum(diag_data, axis=-1)
            diag_data = diag_data / 2.0
            diag_data = tf.expand_dims(diag_data, axis=-1)
            last_dims_t = (3,)
            attention_dims_t = (2,)
            if is_query:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=-1,
                                                                                            keepdims=True)) + numerical_stabilizer)
            else:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=[-2, -1],
                                                                                            keepdims=True)) + numerical_stabilizer)
            return data_dash

        if feature_map == 'relu':
            Q_prime = relu_kernel_transformation(Q, True, projection_matrix)
            K_prime = relu_kernel_transformation(K, False, projection_matrix)
        elif feature_map == 'softmax':
            Q_prime = softmax_kernel_transformation(Q, D_qk, True, projection_matrix)
            K_prime = softmax_kernel_transformation(K, D_qk, False, projection_matrix)
        else:
            raise NotImplementedError("Not Implemented Yet")

        Q_prime = tf.transpose(Q_prime, [1, 0, 2, 3])
        K_prime = tf.transpose(K_prime, [1, 0, 2, 3])
        V = tf.transpose(V, [1, 0, 2, 3])
        if use_recency_bias: gate = tf.transpose(gate, [1, 0, 2])

        @tf.custom_gradient
        def causal_numerator(qs, ks, vs, gate=None):
            result = []
            sums = tf.zeros([B, H, D_map, D_out], dtype=dtype)
            if gate is None:
                for i_th in range(L_out):
                    sums = sums + tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(L_out):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                grads = tf.zeros([B, H, D_map, D_out], dtype=dtype)

                gradient_sums = sums

                q_grads = []
                k_grads = []
                v_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhmo,bho->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo", qs[i_th], res_grad[i_th])
                        k_grads.append(tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[None, Ellipsis])
                        v_grads.append(tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[None, Ellipsis])
                        gradient_sums = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads

                else:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhmo,bho->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo", qs[i_th], res_grad[
                            i_th])  # (1 - gate[i_th]) * bhmo(from bhm * bho) + gate[i_th] * sums[i_th-1]

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]  # bhm
                        v_grads_gate = tf.gradients(gate[i_th], vs[i_th])[0]  # bho

                        k_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[
                            None, Ellipsis]
                        v_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[
                            None, Ellipsis]

                        gradient_sums = gradient_sums - (
                                (1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                        gradient_sums = gradient_sums / gate[i_th]

                        sums_before_minus_curr_cross = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                        if k_grads_gate is None:
                            k_grads.append(k_grads_k_cross_v)
                        else:
                            k_grads_with_gate = k_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-1)
                            # TODO not sure that k_grads_gate has shape of bho, and not sure einsum is sufficient here
                            k_grads.append(k_grads_with_gate + k_grads_k_cross_v)

                        if v_grads_gate is None:
                            v_grads.append(v_grads_k_cross_v)
                        else:
                            v_grads_with_gate = v_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-2)
                            # TODO not sure that v_grads_gate has shape of bhm, and not sure einsum is sufficient here
                            v_grads.append(v_grads_with_gate + v_grads_k_cross_v)

                        gate_grads_term = tf.gradients(tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]), gate[i_th])[0]
                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_cross)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term +
                                              tf.reduce_sum(sums_before_minus_curr_cross, axis=[-2, -1])[
                                                  Ellipsis, None])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads, gate_grads

            return result, grad

        @tf.custom_gradient
        def causal_denominator(qs, ks, gate=None):
            result = []
            sums = tf.zeros([B, H, D_map])
            if gate is None:
                for i_th in range(L_out):
                    sums = sums + ks[i_th]
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(L_out):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * ks[i_th])
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                z_grad = tf.zeros([B, H, D_map])

                gradient_sums = sums

                q_grads = []
                k_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])
                        k_grads.append(z_grad[None, Ellipsis])
                        gradient_sums = gradient_sums - ks[i_th]

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], aixs=0)

                    return q_grads, k_grads

                else:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]

                        gradient_sums = gradient_sums - ((1.0 - gate[i_th]) * ks[i_th])
                        gradient_sums = gradient_sums / gate[i_th]

                        sums_before_minus_curr_k = gradient_sums - ks[i_th]

                        if k_grads_gate is None:
                            k_grads.append((1.0 - gate[i_th]) * z_grad)
                        else:
                            k_grads.append(k_grads_gate * sums_before_minus_curr_k + z_grad)

                        gate_grads_term = tf.gradients(ks[i_th], gate[i_th])[0]

                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_k)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term + sums_before_minus_curr_k)

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, gate_grads

            return result, grad

        def noncausal_numerator(qs, ks, vs, gate=None):
            if gate is None:
                kvs = tf.einsum("lbhm,lbho->bhmo", ks, vs)
                return tf.einsum("Lbhm,bhmo->Lbho", qs, kvs)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                kvs = tf.einsum("lbhm,lbho->lbhmo", ks, vs)
                remainder_kvs = tf.einsum("lbh,lbhmo->bhmo", reverse_gate, kvs)
                fully_gated_kvs = tf.einsum("lbh,bhmo->lbhmo", gate, remainder_kvs) - tf.einsum("lbh,lbhmo->lbhmo",
                                                                                                tf.math.square(
                                                                                                    reverse_gate), kvs)
                return tf.einsum("Lbhm,Lbhmo->Lbho", qs, fully_gated_kvs)

        def noncausal_denominator(qs, ks, gate=None):
            if gate is None:
                ks_sum = tf.einsum("lbhm,l->bhm", ks, tf.ones([L], dtype=dtype))
                return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                remainder_ks = tf.einsum("lbh,lbhm->bhm", reverse_gate, ks)
                fully_gated_ks = tf.einsum("lbh,bhm->lbhm", gate, remainder_ks) - tf.einsum("lbh,lbhm->lbhm",
                                                                                            tf.math.square(
                                                                                                reverse_gate), ks)
                return tf.einsum("Lbhm,Lbhm->Lbh", qs, fully_gated_ks)

        if input_relation == 'causal':
            av_attention = causal_numerator(Q_prime, K_prime, V, gate=gate)
            attention_normalizer = causal_denominator(Q_prime, K_prime, gate=gate)
        else:
            av_attention = noncausal_numerator(Q_prime, K_prime, V, gate=gate)
            attention_normalizer = noncausal_denominator(Q_prime, K_prime, gate=gate)

        out_ = av_attention / attention_normalizer[Ellipsis, None]
        out_ = tf.transpose(out_, [1, 0, 2, 3])

        def restore_projection(xs, units, restore_units, length, mask, name='restore'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                              projector_dilations))
                X = convolution(tf.reshape(xs, [B, length, units]), 1, restore_units, projector_kernel_size,
                                projector_strides, projector_dilations,
                                padding=('valid' if input_relation == 'causal' else 'same'),
                                mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias,
                                activation=projector_activation,
                                weight_initializer=projector_weight_initializer,
                                weight_regularizer=projector_weight_regularizer,
                                weight_constraint=projector_weight_constraint,
                                weight_normalizer=projector_weight_normalizer,
                                bias_initializer=projector_bias_initializer,
                                bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint,
                                trainable=trainable, scope='projection_' + name)
            elif projector == 'dense':
                X = dense(tf.reshape(xs, [B, length, units]), restore_units, mask=mask, groups=projector_groups,
                          use_bias=projector_use_bias, activation=projector_activation,
                          weight_initializer=projector_weight_initializer,
                          weight_regularizer=projector_weight_regularizer,
                          weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                          bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer,
                          bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_' + name)
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_' + name):
                    kernel = tf.get_variable('kernel', shape=[H, units // H, restore_units], dtype=dtype,
                                             initializer=projector_weight_initializer,
                                             regularizer=projector_weight_regularizer, trainable=trainable,
                                             constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: kernel = projector_weight_normalizer(kernel)
                    X = tf.einsum("blho,hor->blr", xs, kernel)
                    if projector_use_bias:
                        bias = tf.get_variable('bias', shape=[restore_units], dtype=dtype,
                                               initializer=projector_bias_initializer,
                                               regularizer=projector_bias_regularizer, trainable=trainable,
                                               constraint=projector_bias_constraint)
                        X = X + bias[None, None]
            else:
                X = tf.reshape(xs, [B, length, restore_units])
            return X

        out_ = restore_projection(out_, D_out, D_out, L_out, mask_query, name='restore')
        out_ = tf.reshape(out_, Bs_q + [L_out, D_out])
        if activation: out_ = activation(out_)

    return out_


def modulated_favorplus(queries,
                        keys,
                        values,
                        style=None,
                        units=None,
                        map_units=None,
                        heads=4,
                        input_relation='self',
                        demodulate=True,
                        fused=True,
                        feature_map='relu',
                        feature_projection=True,
                        mask_query=None,
                        mask_key=None,
                        mask_value=None,
                        cache=None,
                        use_recency_bias=False,
                        activation=None,
                        seed_value=None,
                        projector='einsum_dense',
                        projector_kernel_size=3,
                        projector_strides=1,
                        projector_dilations=None,
                        projector_mask_query=None,
                        projector_mask_key=None,
                        projector_mask_value=None,
                        projector_groups=1,
                        projector_use_bias=True,
                        projector_activation=None,
                        projector_weight_normalizer=None,
                        projector_gain=1,
                        projector_use_wscale=True,
                        projector_lrmul=1,
                        trainable=True,
                        scope='favorplus'):
    if not isinstance(heads, int): raise ValueError(
        "The number of heads must be integer, but given {}".format(type(heads)))

    *Bs_q, L_out, D_q = shape_of(queries)
    *Bs, L, D_k = shape_of(keys)
    *Bs_v, L_v, D_v = shape_of(values)

    D_qk = max(D_q, D_k)

    H = heads if isinstance(heads, int) else 1

    D_out = units if isinstance(units, int) else D_q
    if D_out % H != 0: raise ValueError(
        "The number of heads must divide units evenly, but heads:{} and units: {}".format(H, D_out))

    D_map = map_units if isinstance(map_units, int) else D_qk // 2
    if D_map % H != 0: raise ValueError(
        "The number of heads must divide map units (half of score units)  evenly, but heads:{} and map units: {}".format(
            H, D_map))

    if not input_relation in ['self', 'causal', 'cross']: raise ValueError(
        "input relation must be one of `self`, `causal`, `cross`.")
    if not feature_map in ['relu', 'softmax']: raise NotImplementedError(
        "given method {} of shift-invariant kernels is unsupported.".format(feature_map))

    with tf.variable_scope(scope):
        graph_name = tf.get_default_graph().get_name_scope()
        dtype = queries.dtype
        B = tf.reduce_prod(Bs, keepdims=False)

        # batch shape change
        queries, keys, values = [tf.reshape(in_, [B, length, numerics]) for in_, length, numerics in
                                 [(queries, L_out, D_q), (keys, L, D_k), (values, L, D_v)]]

        mask_query = tf.ones([B, L_out], dtype=dtype) if mask_query is None else tf.reshape(mask_query, [B, L_out])
        mask_key = tf.ones([B, L], dtype=dtype) if mask_key is None else tf.reshape(mask_key, [B, L])
        mask_value = tf.ones([B, L], dtype=dtype) if mask_value is None else tf.reshape(mask_value, [B, L])

        if projector_mask_query is not None: projector_mask_query = tf.reshape(projector_mask_query, [B, L_out])
        if projector_mask_key is not None: projector_mask_key = tf.reshape(projector_mask_key, [B, L])
        if projector_mask_value is not None: projector_mask_value = tf.reshape(projector_mask_value, [B, L])

        def get_weight(shape, gain=1, use_wscale=True, lrmul=1, trainable=True, weight_name='weight'):
            fan_in = np.prod(shape[:-1], dtype=np.float32)
            he_std = gain / np.sqrt(fan_in)
            if use_wscale:
                init_std = 1.0 / lrmul
                runtime_coef = he_std * lrmul
            else:
                init_std = he_std / lrmul
                runtime_coef = lrmul
            return tf.get_variable(weight_name, shape=shape, initializer=tf.random_normal_initializer(0, init_std),
                                   trainable=trainable) * runtime_coef

        def modulated_projection(xs, style, units, material_units, length, mask, name='X'):

            with tf.variable_scope(name):
                if style is not None:
                    style_shape = shape_of(style)
                    if len(style_shape) > 2:
                        style = tf.reshape(style, [style_shape[0], tf.reduce_prod(style_shape[1:])])
                        style_shape = shape_of(style)
                    S = dense(style, units, trainable=trainable, scope='style')

                if projector == 'convolution':
                    if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size,
                                                                                  projector_dilations))
                    weight = get_weight(
                        [projector_kernel_size, units, material_units], projector_gain, projector_use_wscale,
                        projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)
                    weight_new = weight[None] if style is None else weight[None] * S[:, None, :, None]

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('bkdm,bkdm->bm', weight_new, weight_new) + 1e-16)
                        weight_new *= demodulator[:, None, None]

                    if fused:
                        xs = tf.reshape(tf.transpose(xs, [1, 0, 2]), [length, -1])[None]
                        weight = tf.reshape(tf.transpose(weight_new, [1, 2, 0, 3]),
                                            [projector_kernel_size, units, B * material_units])
                        X = tf.nn.conv2d(xs[:, None], weight[None],
                                         padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                         data_format='NHWC')
                        X = tf.transpose(tf.reshape(X[0], [length, B, H, material_units // H]), [1, 0, 2, 3])
                    elif style is None:
                        X = tf.nn.conv2d(xs[:, None], weight_new,
                                         padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                         data_format='NHWC')
                        X = tf.reshape(X[:, 0], [B, length, H, material_units // H])
                    else:
                        xs = tf.transpose(xs, [1, 2, 0])[None]
                        weight = tf.transpose(weight_new, [1, 2, 0, 3])
                        X = tf.nn.depthwise_conv2d(xs, weight, strides=[1, 1, 1, 1],
                                                   padding=('VALID' if input_relation == 'causal' else 'SAME'),
                                                   data_format='NHWC')
                        X = tf.transpose(tf.reshape(X, [length, B, H, material_units // H]), [1, 0, 2, 3])

                elif projector == 'einsum_dense':
                    weight = get_weight(
                        [units, H, material_units // H], projector_gain, projector_use_wscale, projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)
                    weight_new = weight if style is None else weight[None] * S[Ellipsis, None, None]

                    if demodulate:
                        if style is None:
                            demodulator = tf.math.rsqrt(tf.einsum('dhm,dhm->hm', weight_new, weight_new) + 1e-16)
                            weight_new *= demodulator[None]
                        else:
                            demodulator = tf.math.rsqrt(tf.einsum('bdhm,bdhm->bhm', weight_new, weight_new) + 1e-16)
                            weight_new *= demodulator[:, None]

                    if fused:
                        xs = tf.transpose(xs, [1, 2, 0])
                        weight = tf.transpose(weight_new, [1, 0, 2, 3])
                        X = tf.transpose(tf.einsum('lcb,cbhd->lbhd', xs, weight), [1, 0, 2, 3])
                    elif style is None:
                        X = tf.einsum('blc,chd->blhd', xs, weight_new)
                    else:
                        X = tf.einsum('blc,bchd->blhd', xs, weight_new)

                else:
                    X = tf.reshape(xs, [B, length, H, material_units // H])

                if projector_use_bias:
                    X += tf.get_variable('bias', [material_units // H], dtype=dtype, initializer=tf.zeros_initializer(),
                                         trainable=trainable)[None, None, None]
                if projector_activation: X = projector_activation(X)
            return X

        # material projection
        Q = modulated_projection(queries, style, D_q, D_qk, L_out, projector_mask_query, name='Q')
        K = modulated_projection(keys, style, D_k, D_qk, L, projector_mask_key, name='K')
        V = modulated_projection(values, style, D_v, D_out, L, projector_mask_value, name='V')

        # projection
        def _create_products_of_givens_rotations(d, seed=None):
            n_givens_rotations = d * int(math.ceil(math.log(float(d))))
            q = np.eye(d, d)
            if seed is not None: np.random.seed(seed)
            for _ in range(n_givens_rotations):
                random_angle = math.pi * np.random.uniform()
                random_indices = np.random.choice(d, 2)
                index_i = min(random_indices[0], random_indices[1])
                index_j = max(random_indices[0], random_indices[1])
                slice_i = q[index_i]
                slice_j = q[index_j]
                new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
                new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
                q[index_i] = new_slice_i
                q[index_j] = new_slice_j
            return tf.cast(tf.constant(q), dtype=dtype)

        def _create_projection_matrix(m, d, seed, scaling=0, struct_mode=False):
            n_full_blocks = m // d
            block_list = []
            current_seed = seed
            for _ in range(n_full_blocks):
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q)
                current_seed += 1
            remaining_rows = m - n_full_blocks * d
            if remaining_rows > 0:
                if struct_mode:
                    q = _create_products_of_givens_rotations(d, seed)
                else:
                    unstructured_block = tf.random.normal((d, d), seed=current_seed)
                    q, _ = tf.linalg.qr(unstructured_block)
                    q = tf.transpose(q)
                block_list.append(q[0:remaining_rows])
            final_matrix = tf.concat(block_list, axis=0)
            current_seed += 1

            if scaling == 0:
                multiplier = tf.norm(tf.random.normal((m, d), seed=current_seed), axis=1)
            elif scaling == 1:
                multiplier = tf.math.sqrt(float(d)) * tf.ones(float(m))
            else:
                raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)
            return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)

        if feature_projection:
            # seed = tf.math.ceil(tf.math.abs(tf.math.reduce_sum(Q) * 1e8))
            # seed = tf.dtypes.cast(seed, tf.int32)
            # projection_matrix = _create_projection_matrix(seed=seed)
            seed = random.randint(0, (2 ** 31) - 1) if seed_value is None else seed_value
            projection_matrix = _create_projection_matrix(D_map, D_qk // H, seed=seed)
        else:
            projection_matrix = None

        if use_recency_bias:
            gate = dense(keys, H, mask=projector_mask_key, groups=projector_groups, use_bias=True, activation=sigmoid,
                         trainable=trainable, scope='gate_dense')
        else:
            gate = None

        # cache
        if cache:
            K, V = [tf.concat([cache['K'], K], -2), tf.concat([cache['V'], V], -2)]
            if use_recency_bias:
                gate = tf.concat([cache['gate'], gate], -2)
                cache['gate'] = gate
            cache_shape = shape_of(cache['K'])
            L = cache_shape[-2] + L
            mask_key = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_key], axis=-2)
            mask_value = tf.concat([tf.ones(cache_shape[:-1], dtype=dtype), mask_value], axis=-2)
            cache['K'], cache['V'] = K, V
            input_relation = 'cross'

        def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):

            del is_query
            if projection_matrix is None:
                return relu(data) + numerical_stabilizer
            else:
                ratio = 1.0 * tf.math.rsqrt(tf.cast(D_map, tf.float32))
                data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
                return relu(data_dash) + numerical_stabilizer

        def softmax_kernel_transformation(data, units_data, is_query, projection_matrix=None,
                                          numerical_stabilizer=0.000001):

            data_normalizer = 1.0 * tf.math.rsqrt(tf.math.sqrt(tf.cast(units_data, tf.float32)))
            data = data_normalizer * data

            ratio = 1.0 * tf.math.rsqrt(tf.cast(D_map, tf.float32))
            data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
            diag_data = tf.math.square(data)
            diag_data = tf.math.reduce_sum(diag_data, axis=-1)
            diag_data = diag_data / 2.0
            diag_data = tf.expand_dims(diag_data, axis=-1)
            last_dims_t = (3,)
            attention_dims_t = (2,)
            if is_query:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=-1,
                                                                                            keepdims=True)) + numerical_stabilizer)
            else:
                data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash, axis=[-2, -1],
                                                                                            keepdims=True)) + numerical_stabilizer)
            return data_dash

        if feature_map == 'relu':
            Q_prime = relu_kernel_transformation(Q, True, projection_matrix)
            K_prime = relu_kernel_transformation(K, False, projection_matrix)
        elif feature_map == 'softmax':
            Q_prime = softmax_kernel_transformation(Q, D_qk, True, projection_matrix)
            K_prime = softmax_kernel_transformation(K, D_qk, False, projection_matrix)
        else:
            raise NotImplementedError("Not Implemented Yet")

        Q_prime = tf.transpose(Q_prime, [1, 0, 2, 3])
        K_prime = tf.transpose(K_prime, [1, 0, 2, 3])
        V = tf.transpose(V, [1, 0, 2, 3])
        if use_recency_bias: gate = tf.transpose(gate, [1, 0, 2])

        @tf.custom_gradient
        def causal_numerator(qs, ks, vs, gate=None):
            result = []
            sums = tf.zeros([B, H, D_map, D_out], dtype=dtype)
            if gate is None:
                for i_th in range(L_out):
                    sums = sums + tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(L_out):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                    result.append(tf.einsum("bhmo,bhm->bho", sums, qs[i_th])[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                grads = tf.zeros([B, H, D_map, D_out], dtype=dtype)

                gradient_sums = sums

                q_grads = []
                k_grads = []
                v_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhmo,bho->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo", qs[i_th], res_grad[i_th])
                        k_grads.append(tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[None, Ellipsis])
                        v_grads.append(tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[None, Ellipsis])
                        gradient_sums = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads

                else:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhmo,bho->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        grads = grads + tf.einsum("bhm,bho->bhmo", qs[i_th], res_grad[
                            i_th])  # (1 - gate[i_th]) * bhmo(from bhm * bho) + gate[i_th] * sums[i_th-1]

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]  # bhm
                        v_grads_gate = tf.gradients(gate[i_th], vs[i_th])[0]  # bho

                        k_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bho->bhm", grads, vs[i_th])[
                            None, Ellipsis]
                        v_grads_k_cross_v = (1.0 - gate[i_th]) * tf.einsum("bhmo,bhm->bho", grads, ks[i_th])[
                            None, Ellipsis]

                        gradient_sums = gradient_sums - (
                                (1.0 - gate[i_th]) * tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]))
                        gradient_sums = gradient_sums / gate[i_th]

                        sums_before_minus_curr_cross = gradient_sums - tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th])

                        if k_grads_gate is None:
                            k_grads.append(k_grads_k_cross_v)
                        else:
                            k_grads_with_gate = k_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-1)
                            # TODO not sure that k_grads_gate has shape of bho, and not sure einsum is sufficient here
                            k_grads.append(k_grads_with_gate + k_grads_k_cross_v)

                        if v_grads_gate is None:
                            v_grads.append(v_grads_k_cross_v)
                        else:
                            v_grads_with_gate = v_grads_gate * tf.reduce_sum(sums_before_minus_curr_cross, axis=-2)
                            # TODO not sure that v_grads_gate has shape of bhm, and not sure einsum is sufficient here
                            v_grads.append(v_grads_with_gate + v_grads_k_cross_v)

                        gate_grads_term = tf.gradients(tf.einsum("bhm,bho->bhmo", ks[i_th], vs[i_th]), gate[i_th])[0]
                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_cross)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term +
                                              tf.reduce_sum(sums_before_minus_curr_cross, axis=[-2, -1])[
                                                  Ellipsis, None])

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    v_grads = tf.concat(v_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, v_grads, gate_grads

            return result, grad

        @tf.custom_gradient
        def causal_denominator(qs, ks, gate=None):
            result = []
            sums = tf.zeros([B, H, D_map])
            if gate is None:
                for i_th in range(L_out):
                    sums = sums + ks[i_th]
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            else:
                gate = gate[Ellipsis, None]
                for i_th in range(L_out):
                    sums = (gate[i_th] * sums) + ((1.0 - gate[i_th]) * ks[i_th])
                    result.append(tf.reduce_sum(qs[i_th] * sums, axis=2)[None, Ellipsis])
            result = tf.concat(result, axis=0)

            def grad(res_grad):
                z_grad = tf.zeros([B, H, D_map])

                gradient_sums = sums

                q_grads = []
                k_grads = []
                if gate is not None:
                    gate_grads = []

                if gate is None:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])
                        k_grads.append(z_grad[None, Ellipsis])
                        gradient_sums = gradient_sums - ks[i_th]

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], aixs=0)

                    return q_grads, k_grads

                else:
                    for i_th in range(L_out - 1, -1, -1):
                        q_grads.append(
                            tf.einsum("bhm,bh->bhm", gradient_sums, res_grad[i_th])[None, Ellipsis])
                        z_grad = z_grad + tf.einsum("bhm,bh->bhm", qs[i_th], res_grad[i_th])

                        k_grads_gate = tf.gradients(gate[i_th], ks[i_th])[0]

                        gradient_sums = gradient_sums - ((1.0 - gate[i_th]) * ks[i_th])
                        gradient_sums = gradient_sums / gate[i_th]

                        sums_before_minus_curr_k = gradient_sums - ks[i_th]

                        if k_grads_gate is None:
                            k_grads.append((1.0 - gate[i_th]) * z_grad)
                        else:
                            k_grads.append(k_grads_gate * sums_before_minus_curr_k + z_grad)

                        gate_grads_term = tf.gradients(ks[i_th], gate[i_th])[0]

                        if gate_grads_term is None:
                            gate_grads.append(sums_before_minus_curr_k)
                        else:
                            gate_grads.append((1.0 - gate[i_th]) * gate_grads_term + sums_before_minus_curr_k)

                    q_grads = tf.concat(q_grads[::-1], axis=0)
                    k_grads = tf.concat(k_grads[::-1], axis=0)
                    gate_grads = tf.concat(gate_grads[::-1], axis=0)

                    return q_grads, k_grads, gate_grads

            return result, grad

        def noncausal_numerator(qs, ks, vs, gate=None):
            if gate is None:
                kvs = tf.einsum("lbhm,lbho->bhmo", ks, vs)
                return tf.einsum("Lbhm,bhmo->Lbho", qs, kvs)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                kvs = tf.einsum("lbhm,lbho->lbhmo", ks, vs)
                remainder_kvs = tf.einsum("lbh,lbhmo->bhmo", reverse_gate, kvs)
                fully_gated_kvs = tf.einsum("lbh,bhmo->lbhmo", gate, remainder_kvs) - tf.einsum("lbh,lbhmo->lbhmo",
                                                                                                tf.math.square(
                                                                                                    reverse_gate), kvs)
                return tf.einsum("Lbhm,Lbhmo->Lbho", qs, fully_gated_kvs)

        def noncausal_denominator(qs, ks, gate=None):
            if gate is None:
                ks_sum = tf.einsum("lbhm,l->bhm", ks, tf.ones([L], dtype=dtype))
                return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)
            else:
                # not for cross attention
                reverse_gate = 1.0 - gate
                remainder_ks = tf.einsum("lbh,lbhm->bhm", reverse_gate, ks)
                fully_gated_ks = tf.einsum("lbh,bhm->lbhm", gate, remainder_ks) - tf.einsum("lbh,lbhm->lbhm",
                                                                                            tf.math.square(
                                                                                                reverse_gate), ks)
                return tf.einsum("Lbhm,Lbhm->Lbh", qs, fully_gated_ks)

        if input_relation == 'causal':
            av_attention = causal_numerator(Q_prime, K_prime, V, gate=gate)
            attention_normalizer = causal_denominator(Q_prime, K_prime, gate=gate)
        else:
            av_attention = noncausal_numerator(Q_prime, K_prime, V, gate=gate)
            attention_normalizer = noncausal_denominator(Q_prime, K_prime, gate=gate)

        out_ = av_attention / attention_normalizer[Ellipsis, None]
        out_ = tf.transpose(out_, [1, 0, 2, 3])
        out_ = out_ - (tf.einsum('blhc->bhc', out_) / tf.einsum('bl->b', mask_query)[:, None, None])[:, None]
        out_ = out_ * mask_query[Ellipsis, None, None]

        '''
        def restore_projection(xs, units, restore_units, length, mask, name='restore'):
            if projector == 'convolution':
                if input_relation == 'causal': xs = tf.pad(xs, causal_padding(xs, projector_kernel_size, projector_dilations))
                X = convolution(tf.reshape(xs, [B, length, units]), 1, restore_units, projector_kernel_size, projector_strides, projector_dilations, padding=('valid' if input_relation=='causal' else 'same'),
                    mask_in=mask, mask_out=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_initializer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name)
            elif projector == 'dense':
                X = dense(tf.reshape(xs, [B, length, units]), restore_units, mask=mask, groups=projector_groups, use_bias=projector_use_bias, activation=projector_activation,
                    weight_initializer=projector_weight_initializer, weight_regularizer=projector_weight_regularizer, weight_constraint=projector_weight_constraint, weight_normalizer=projector_weight_normalizer,
                    bias_iniaitlizer=projector_bias_initializer, bias_regularizer=projector_bias_regularizer, bias_constraint=projector_bias_constraint, trainable=trainable, scope='projection_'+name)
            elif projector == 'einsum_dense':
                with tf.variable_scope('projection_'+name):
                    kernel = tf.get_variable('kernel', shape=[H, units//H, restore_units], dtype=dtype,
                            initializer=projector_weight_initializer, regularizer=projector_weight_regularizer, trainable=trainable, constraint=projector_weight_constraint)
                    if projector_weight_normalizer is not None: kernel = projector_weight_normalizer(kernel)
                    X = tf.einsum("blho,hor->blr", xs, kernel)
            else: raise NotImplementedError
            return X
        '''

        def modulated_restoration(xs, units, restore_units, mask, name='O'):

            with tf.variable_scope(name):
                if projector == 'convolution':
                    if input_relation == 'causal':
                        xs = tf.pad(xs, causal_padding(xs, projector_kernel_size, projector_dilations))
                    else:
                        same_pad = projector_kernel_size // 2
                        xs = tf.pad(xs, [[0, 0], [same_pad, same_pad], [0, 0], [0, 0]])

                    weight = get_weight(
                        [projector_kernel_size, H, units, restore_units], projector_gain, projector_use_wscale,
                        projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('khmo,khmo->o', weight, weight) + 1e-16)
                        weight *= demodulator[None, None, None]

                    X = tf.nn.conv2d(xs, weight, padding='VALID', data_format='NHWC')
                    X = tf.squeeze(X, axis=2)

                elif projector == 'einsum_dense':
                    weight = get_weight(
                        [H, units, restore_units], projector_gain, projector_use_wscale, projector_lrmul,
                        trainable=trainable, weight_name='weight')
                    if projector_weight_normalizer: weight = projector_weight_normalizer(weight)

                    if demodulate:
                        demodulator = tf.math.rsqrt(tf.einsum('hmo,hmo->o', weight, weight) + 1e-16)
                        weight *= demodulator[None, None]

                    X = tf.einsum('blhm,hmo->blo', xs, weight)

                else:
                    X = tf.reshape(xs, [B, L_out, restore_units])

                if projector_use_bias:
                    X += tf.get_variable('bias', [restore_units], dtype=dtype, initializer=tf.zeros_initializer(),
                                         trainable=trainable)[None, None]
                if mask is not None: X *= mask[Ellipsis, None]
            return X

        out_ = modulated_restoration(out_, D_out // H, D_out, mask_query, name='restore')
        out_ = tf.reshape(out_, Bs_q + [L_out, D_out])

    return out_
