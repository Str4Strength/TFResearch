import os
import math
import warnings
#import operator

import scipy.signal

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from functools import partial
from matplotlib.cm import get_cmap
from termcolor import cprint, colored



def list_elemops(x, y=None, mode='+'):
    if mode in ['+', '-', '*', '/']: assert y
    if mode=='+':
        return list(map(lambda a,b: a+b, x, y))
    if mode=='-':
        return list(map(lambda a,b: a-b, x, y))
    if mode=='*':
        return list(map(lambda a,b: a*b, x, y))
    if mode=='/':
        return list(map(lambda a,b: a/b, x, y))
    if mode=='//':
        return list(map(lambda a,b: a//b, x, y))
    return list(map(lambda a: mode(a), x))
    raise ValueError

def list_elemops_iter(inputs, modes=[None]):
    assert len(inputs)-1 == len(modes)
    output = inputs[0]
    for n, mode in enumerate(modes):
        output = list_elemops(output, inputs[n+1], mode=mode)
    return output

def order_matching(down_ratios, up_ratios):
    R_rec = list(map(np.cumprod, zip(*down_ratios[::-1])))
    R_mul = list(map(np.cumprod, zip(*up_ratios)))
    #print(R_rec[0]) #print(R_rec[1]) #print(R_mul[0]) #print(R_mul[1])
    #cprint((R_rec, R_mul), 'cyan')
    match_h=[]
    match_w=[]
    for n in range(len(R_mul[0])):
        h_ = np.array([R_mul[0][n] > RRH for RRH in R_rec[0]]).astype(np.int32)
        w_ = np.array([R_mul[1][n] > RRW for RRW in R_rec[1]]).astype(np.int32)
        #cprint((n, h_, w_), 'cyan')
        match_h.append(min(np.sum(h_), len(R_rec[0])-1))
        match_w.append(min(np.sum(w_), len(R_rec[1])-1))
    #print(match_h) #print(match_w)
    matched_order = np.maximum(match_h, match_w)
    ups_h, ups_w, downs_h, downs_w = ([],[],[],[])
    for i, o in enumerate(matched_order):
        ups_h.append(max(1, R_mul[0][i] // R_rec[0][o]))
        ups_w.append(max(1, R_mul[1][i] // R_rec[1][o]))
        downs_h.append(max(1, R_rec[0][o] // R_mul[0][i]))
        downs_w.append(max(1, R_rec[1][o] // R_mul[1][i]))
    #print(ups_h) #print(ups_w) #print(downs_h) #print(downs_w)
    match_infos = list(zip(*[matched_order, ups_h, downs_h, ups_w, downs_w]))
    #print(list(zip(*R_mul)))
    #via_rec = [[R_rec[0][o] * ups_h[n] // downs_h[n], R_rec[1][o] * ups_w[n] // downs_w[n]]
    #        for n, o in enumerate(matched_order)]
    #print(via_rec)
    #cprint(match_infos, 'cyan')
    return match_infos

def colorize(value, vmin=None, vmax=None, reverse=True, transpose=True, cmap='viridis'):
    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = tf.clip_by_value((value - vmin) / (vmax - vmin), 0, 1)

    value = tf.reverse(value, [-1]) if reverse else value
    value = tf.transpose(value, [0, 2, 1]) if transpose else value

    # quantize
    indices = tf.cast(tf.round(value * 255), tf.int32)

    # gather
    cm = get_cmap(cmap)
    colors = tf.constant(cm(range(256)), dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def tf_print(name, value, summary=True, color='yellow'):
    return tf.print(
            colored("{}".format(name), color),
            tf.shape(value),
            tf.reduce_min(value), '~',
            tf.reduce_mean(value), '~',
            tf.reduce_max(value),
            "\n" if not summary else '',
            value if not summary else '',
            summarize=256)

def print_list(list_, color=None):
    if len(list_)==0:
        return
    cols = os.get_terminal_size().columns
    max_list = max(map(len, list_))
    new_list = map(lambda e: colored(e+' '*(max_list-len(e)), color), list_)
    ne_line = cols//(max_list+1)
    for idx,e in enumerate(new_list):
        end='\n' if (idx+1)%ne_line==0 else ''
        print(f'{e} ', end=end)
    print('')

