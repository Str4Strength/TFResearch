import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

from tqdm import tqdm
from functools import partial
from termcolor import cprint

from tensorpack.dataflow import DataFromList, MultiProcessMapAndBatchDataZMQ
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.sessinit import ChainInit, SaverRestore
from tensorpack.callbacks import JSONWriter, ScalarPrinter, TFEventWriter
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.train.interface import TrainConfig, launch_train_with_config

from ..segmentation import *
from ..hyper_params import hyper_params as hp

from .meta import *

import os
pj = os.path.join
import json
import time
import random


import cv2
from pycocotools.coco import COCO



def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"

class DataFeeder():
    def __init__(
            self,
            gpus,
            data_dir,
            mode = 'train_all',
            batch_size = 1,
            hot_mode = True,
            transform = None
            ):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.coco = COCO(pj(data_dir, mode + '.json'))
        self.B = batch_size
        self.hot_mode = hot_mode
        self.hot_codes = np.eye(len(self.coco.getCatIds()) + 1)

        with open(pj(data_dir, f'{mode}.json'), 'r') as f:
            dataset = json.loads(f.read())

        category_names = {}
        for cat_it in dataset['categories']: category_names[cat_it['id']] = cat_it['name']

        self.value_to_cat = category_names.copy()
        self.cat_to_value = dict(zip(category_names.values(), category_names.keys()))



    def _get_input_(self, index: int):
        image_id = self.coco.getImgIds(imgIds = index)
        image_infos = self.coco.loadImgs(image_id)[0]

        images = cv2.imread(pj(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ('train', 'train_all', 'val'):
            ann_ids = self.coco.getAnnIds(imgIds = image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key = lambda ids: len(ids['segmentation'][0]), reverse = False)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.cat_to_value[className]
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
            if self.hot_mode: masks = self.hot_codes[masks]

            if exists(self.transform):
                transformed = self.transform(image = images, mask = masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks#, image_infos

        if self.mode == 'test':
            if exists(self.transform):
                transformed = self.transform(image = images)
                images = transformed["image"]
            return images, image_infos

    def __call__(
            self,
            threads = 32,
            buffers = 2,
            ):
        total_epoch = self.coco.getImgIds()
        if len(total_epoch) % self.B != 0:
            original_epoch = total_epoch
            for _ in range(self.B - 1): total_epoch += original_epoch

        minibatches = [total_epoch[id_: id_ + self.B] for id_ in range(0, len(total_epoch), self.B)]

        dataset = DataFromList(minibatches, shuffle = True)
        dataset = MultiProcessMapAndBatchDataZMQ(
                ds = dataset,
                num_proc = threads,
                map_func = self._get_input_,
                batch_size = self.B,
                buffer_size = buffers
                )

        return dataset, (len(self.coco.getImgIds()), len(total_epoch))



class Single_Machine_Trainer(Single_GPU_Trainer, MakeGetOperationFunction):
    def _make_get_op_fn(
            self,
            input,
            get_cost_fn,
            get_opt_fn
            ):
        return MakeGetOperationFunction._make_get_op_fn(input, self, get_cost_fn, get_opt_fn)

class Multi_Machine_Trainer(Batch_Expand_Multi_GPU_Trainer, MakeGetOperationFunction):
    def _make_get_op_fn(
            self,
            input,
            get_cost_fn,
            get_opt_fn
            ):
        return MakeGetOperationFunction._make_get_op_fn(input, self, get_cost_fn, get_opt_fn)


class Solver(ModelDesc):
    def __init__(self, image_size = (512, 512), label_size = 11, **model_kwargs):
        self.global_step = tf.train.get_or_create_global_step()
        self.model_kwargs = model_kwargs
        #self.optimizer = optimizer
        self.W, self.H = image_size[0], image_size[1]
        self.label_size = label_size

    def _pretrain_(
            self,
            image,
            cardinality,
            trainable = True,
            reuse = tf.AUTO_REUSE
            ):
        fmaps, scores = segmentizer(
                image,
                cardinality = cardinality,
                trainable = trainable,
                reuse = reuse,
                **self.model_kwargs
                )
        return scores, fmaps

    def __call__(
            self,
            image,
            cardinality,
            trainable = True,
            reuse = tf.AUTO_REUSE
            ):
        pred, logit = enhanced_segmentizer(
                image,
                cardinality = cardinality,
                trainable = trainable,
                reuse = reuse,
                **self.model_kwargs
                )
        return pred, logit

    def get_optimizer(self):
        x = tf.maximum(tf.cast(self.global_step, dtype = tf.float32) - hp.learn.decay_begin, 0.)
        x = tf.minimum(x / hp.learn.decay_steps, 1.)
        learning_rate = (hp.learn.learning_rate - hp.learn.end_learning_rate) * ((1. - x) ** hp.learn.power)
        learning_rate += hp.learn.end_learning_rate
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = AdamWeightDecayOptimizer(learning_rate = learning_rate, **hp.optimizer)
        return optimizer

    def inputs(self):
        inputs = (
                #(dtype, shape, name)
                (tf.float32, (None, self.W, self.H, 3), 'image'),
                (tf.float32, (None, self.W, self.H, self.label_size), 'label'),
                )
        return list(map(tf.placeholder, *zip(*inputs)))

    def build_graph(
            self,
            images,
            labels,
            ):
        image_presser = tf.range(shape(labels)[-1], dtype = tf.float32) + 1
        image_presser /= tf.reduce_max(image_presser)

        '''
        if hp.train_mode == 'pre':
            logits, fmaps = self._pretrain_(images, cardinality = shape(labels)[-1], trainable = True, reuse = tf.AUTO_REUSE)
            loss = [tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logit) for logit in logits]

            local = locals()
            """
            for n, win_size in enumerate(self.model_kwargs['windows']):
                with tf.name_scope('loss'):
                    local[f'loss_win{win_size:02d}'] = tf.reduce_mean(loss[n])
                    tf.summary.scalar(f'loss_win{win_size:02d}', local[f'loss_win{win_size:02d}'])
                    tf.add_to_collection(tf.GraphKeys.LOSSES, local[f'loss_win{win_size:02d}'])

                with tf.name_scope('visual'):
                    loss_n = loss[n][:1][Ellipsis, None]
                    tf.summary.image(
                            f'loss_map_win{win_size:02d}',
                            loss_n / tf.reduce_max(loss_n),
                            1
                            )
                    tf.summary.image(
                            f'prediction{win_size:02d}',
                            tf.einsum('bhwl,l->bhw', tf.nn.softmax(logits[n][:1], axis = -1), image_presser)[Ellipsis, None],
                            1
                            )
            """
            with tf.name_scope('visual'):
                lossmap = loss[0][:1]
                lossmap /= tf.reduce_max(lossmap)

                tf.summary.image('loss_map',  lossmap[Ellipsis, None], 1)

                pred = tf.nn.softmax(logits[0][:1], axis = -1)
                simple_pred = tf.cast(tf.math.argmax(pred, axis = -1), dtype = pred.dtype) / (shape(labels)[-1] - 1)
                tf.summary.image('prediction', tf.reshape(simple_pred, [1, *shape(labels)[1:-1], 1]), 1)

        elif hp.train_mode == 'post':
            logit = self(images, cardinality = shape(labels)[-1], trainable = True, reuse = tf.AUTO_REUSE)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logit)
            pixelwise_loss_weight = tf.where(
                    tf.equal(labels, 0.0),
                    tf.ones_like(labels, dtype = images.dtype),
                    10.0 * tf.ones_like(labels, dtype = images.dtype)
                    )
            loss *= pixelwise_loss_weight
            loss = tf.reduce_mean(loss)

            local = locals()
            with tf.name_scope('visual'):
                pred = tf.nn.softmax(logit[:1], axis = -1)
                simple_pred = tf.cast(tf.math.argmax(pred, axis = -1), dtype = pred.dtype) / (shape(labels)[-1] - 1)
                tf.summary.image('prediction', tf.reshape(simple_pred, [1, *shape(labels)[1:-1], 1]), 1)
                #calibrate = tf.nn.conv2d(simple_pred, tf.ones([3, 3, 1, 1], dtype = ),
        '''
        local = locals()
        loss_list = []

        pred = enhanced_segmentizer(
                image = images,
                cardinality = shape(labels)[-1],
                trainable = True,
                reuse = tf.AUTO_REUSE,
                **hp.model
                )

        """
        for n, win_size in enumerate(self.model_kwargs['windows']):
            with tf.name_scope(f'loss_{n}'):
                loss_n = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits[n])
                local[f'loss_win{win_size:02d}'] = tf.reduce_mean(loss_n)
                loss_list.append(local[f'loss_win{win_size:02d}'])

                tf.summary.scalar(f'loss_win{win_size:02d}', local[f'loss_win{win_size:02d}'])
                tf.add_to_collection(tf.GraphKeys.LOSSES, local[f'loss_win{win_size:02d}'])

            with tf.name_scope(f'visual_{n}'):
                loss_map_n = loss_n[:1][Ellipsis, None]
                tf.summary.image(f'loss_map_win{win_size:02d}', loss_map_n / tf.reduce_max(loss_map_n), 1)

                segment_n = tf.nn.softmax(logits[n][:1], axis = -1)
                segment_n = tf.cast(tf.argmax(segment_n, axis = -1), dtype = segment_n.dtype) / (shape(labels)[-1] - 1)
                tf.summary.image(f'segment_{win_size:02d}', tf.reshape(segment_n, [1, self.W, self.H, 1]), 1)
        """

        with tf.name_scope('loss'):
            loss_pred = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = pred)
            local['loss_pred'] = tf.reduce_mean(loss_pred)
            loss_list.append(local['loss_pred'] * hp.lambda_main)

            tf.summary.scalar('loss_pred', local['loss_pred'])
            tf.add_to_collection(tf.GraphKeys.LOSSES, local['loss_pred'])

            local['loss_total'] = tf.reduce_sum(loss_list)

            tf.summary.scalar('loss_total', local['loss_total'])
            tf.add_to_collection(tf.GraphKeys.LOSSES, local['loss_total'])

        with tf.name_scope('visual'):
            loss_map_pred = loss_pred[:1][Ellipsis, None]
            tf.summary.image('loss_map_pred', loss_map_pred / tf.reduce_max(loss_map_pred), 1)

            segment_pred = tf.nn.softmax(pred[:1], axis = -1)
            segment_pred = tf.cast(tf.argmax(segment_pred, axis = -1), dtype = segment_pred.dtype) / (shape(labels)[-1] + 1)
            tf.summary.image('segment_pred', tf.reshape(segment_pred, [1, self.W, self.H, 1]), 1)

            tf.summary.image('image', images[:1], 1)
            tf.summary.image('answer', tf.cast(tf.argmax(labels[:1], axis = -1)[Ellipsis, None], dtype = segment_pred.dtype) / (shape(labels)[-1] + 1), 1)
            #tf.einsum('bhwl,l->bhw', labels[:1], image_presser)[Ellipsis, None], 1)

        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, f'^{hp.model.scope}'):
            #if hp.train_mode == 'pre' and f'{hp.model.scope}/scr' in variable.name: continue
            tf.add_to_collection(hp.model.scope, variable)

        returns = ([local['loss_total'],], [[self.model_kwargs['scope'],],])
        #cprint(returns, color = 'cyan')
        return returns


# variable saving에 문제 발생. ModelSaver class 내부 출력 시도
# variable 이 tf.get_collection으로 가져와지지 않는 것처럼 보임. 수동 작업
class ModelSaver2(ModelSaver):
    def _setup_graph(self):
        assert self.checkpoint_dir is not None

        #vars = [tf.get_collection(key) for key in tf.GraphKeys.GLOBAL_STEP]
        vars = []
        for key in self.var_collections:
            #if hp.model.scope not in key: continue
            vars.extend(tf.get_collection(key))

        cprint(f'saving vars: {vars}', color = 'red')

        self.path = pj(self.checkpoint_dir, 'model')
        self.saver = tf.train.Saver(
                var_list = vars,
                max_to_keep = self._max_to_keep,
                keep_checkpoint_every_n_hours = self._keep_every_n_hours,
                write_version = tf.train.SaverDef.V2,
                save_relative_paths = True
                )

        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)



def train(case, gpu):

    num_gpu = len(gpu)
    trainer = partial(Multi_Machine_Trainer, gpus = num_gpu) if num_gpu > 1 else Single_Machine_Trainer
    trainer = trainer(penalty_targets = ["!@#$%"], penalty_rate = [1.0])

    dataset = DataFeeder(
            gpus = num_gpu,
            data_dir = hp.data_dir,
            mode = 'train_all',
            batch_size = hp.batch_size,
            hot_mode = True,
            )
    train_dataflow, (batch_length, even_batch_length) = dataset(threads = hp.threads, buffers = hp.buffers)
    cprint(f'{batch_length}, {-(-batch_length // num_gpu // hp.batch_size)} ---> {even_batch_length}, {even_batch_length // num_gpu // hp.batch_size}', color = 'magenta')

    solver = Solver(image_size = (512, 512), **hp.model)

    #config = tf.ConfigProto()

    save_path = pj(hp.log_dir, case)
    restore_path = pj(hp.log_dir, hp.load_case) if hp.train_mode == 'post' else save_path

    logger.set_logger_dir(save_path)
    monitors = [TFEventWriter(), JSONWriter(), ScalarPrinter()]

    callbacks = []
    for module_name in [hp.model.scope]:
        saver = ModelSaver(
                max_to_keep = 5,
                keep_checkpoint_every_n_hours = 1,
                checkpoint_dir = pj(save_path, module_name),
                var_collections = [tf.GraphKeys.GLOBAL_STEP, module_name],
                )
        callbacks.append(saver)


    current_checkpoint = tf.train.latest_checkpoint(pj(save_path, hp.model.scope))
    if exists(current_checkpoint):
        restore_inits = [
                SaverRestore(
                    model_path = current_checkpoint,
                    prefix = None,
                    #ignore = ['global_step']
                    )
                ]
    else:
        if hp.train_mode == 'post':
            restore_inits = [
                    SaverRestore(
                        model_path = tf.train.latest_checkpoint(pj(restore_path, hp.model.scope)),
                        prefix = None,
                        #ignore = ['global_step', *[f'{hp.model.scope}/scr{win:02d}' for win in hp.model.windows]],
                        ignore = ['global_step'],
                        )
                    ]
        else:
            restore_inits = []


    epoch = JSONWriter.load_existing_epoch_number()

    train_configuration = TrainConfig(
            model = solver,
            dataflow = train_dataflow,
            callbacks = callbacks,
            steps_per_epoch = hp.steps_per_record,
            session_init = ChainInit(restore_inits) if hp.train_mode == 'post' else None,
            starting_epoch = 1,
            monitors = monitors,
            max_epoch = hp.max_epoch
            )

    launch_train_with_config(train_configuration, trainer = trainer)

