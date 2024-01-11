import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import albumentations as A
import pandas as pd

from ..Neural_Network import *
from ..segmentation import *
from ..hyper_params import hyper_params as hp
from ..train.segmentation import *

import os
pj = os.path.join
import time
import cv2

from termcolor import cprint
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



def predict(
        image = None,
        cardinality = 11,
        size = None,
        reuse = tf.AUTO_REUSE,
        session = None
        ):
    image_tensor = tf.placeholder(tf.float32, (None, 512, 512, 3), 'image')

    main = enhanced_segmentizer(
            image = image_tensor,
            cardinality = cardinality,
            trainable = False,
            reuse = reuse,
            **hp.model
            )
    if exists(size): main = tf.image.resize(main, size)

    if exists(image):
        prediction = session.run(main, feed_dict = {image_tensor: image})
        return prediction


class DataLoad(DataFeeder):

    def construct_minibatches(self):
        total_epoch = self.coco.getImgIds()
        self.minibatches = [total_epoch[id_: min(id_ + self.B, len(total_epoch))] for id_ in range(0, len(total_epoch), self.B)]

    #def __call__(self, ids):
    #    dataset = []
    #    for id_ in ids: dataset.append(self._get_input_(id_))
    #    return list(zip(*dataset))

    def __call__(self):
        #datasets = []
        #with ThreadPoolExecutor(max_workers = 16) as pool:
        #    datasets.append(pool.map(self._get_output_, self.minibatches))
        #cprint(datasets, color='red')
        #return list(zip(*datasets))
        return ThreadPoolExecutor(max_workers = 64).map(self._get_output_, self.minibatches)

    def _get_output_(self, ids):
        #dataset = []
        #for ids in self.minibatches:
        images, infos = [], []
        for id_ in ids:
            img, info = self._get_input_(id_)
            images.append(img)
            infos.append(info)
        #with ThreadPoolExecutor(max_workers = 4) as pool:
        #    a = pool.map(self._get_input_, ids)
        #    print(a)
        #    print(a[0], a[1])
        #    images, infos = zip(*a)
            #images.append(img)
            #infos.append(info)
        #dataset.append((images, infos))
        #return dataset
        return images, infos


def test(case, gpu, pred_dir = './submission'):
    os.makedirs(pred_dir, exist_ok = True)

    num_gpu = len(gpu)

    dataset = DataLoad(
            gpus = max(num_gpu, 1),
            data_dir = hp.data_dir,
            mode = 'test',
            batch_size = hp.batch_size,
            )
    dataset.construct_minibatches()
    #test_dataflow, (batch_length, even_test_batch_length) = dataset(threads = hp.threads, buffers = hp.buffers)
    test_dataflow = dataset.minibatches
    cprint((len(dataset.coco.getImgIds()), len(test_dataflow)), color = 'magenta')

    predict(None)
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, f'^.*{hp.model.scope}/')
    #print(variables)
    load = tf.train.Saver(variables)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        global_variables = [v.name[:-2] for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

        latest_checkpoint = tf.train.latest_checkpoint(pj(hp.log_dir, case, hp.model.scope))
        cprint(f'latest checkpoint : {latest_checkpoint}', color = 'yellow')
        reader = tf.train.NewCheckpointReader(latest_checkpoint)
        joint_variables = reader.get_variable_to_shape_map()

        checkpoint_variables = []
        for v in joint_variables:
            if v == 'global_step': continue
            loaded_v = tf.train.load_variable(latest_checkpoint, v)
            global_v = tf.get_default_graph().get_tensor_by_name(v + ':0')

            if loaded_v.shape != global_v.shape:
                print(f'{v:100s}:\n    checkpoint: {str(loaded_v.shape):20s}\n    graph: {str(global_v.shape):20s}')
            checkpoint_variables.append(v)

        load.restore(sess, latest_checkpoint)
        print(f'Successfully loaded the model {latest_checkpoint}')

        print('Only model variables: ')
        for v in [x for x in global_variables if x not in checkpoint_variables]: cprint(v, color = 'red')
        print('Only checkpoint variables: ')
        for v in [x for x in checkpoint_variables if x not in global_variables]: cprint(v, color = 'magenta')


        size = 256
        #transform = A.Compose([A.Resize(size, size)])
        print('Start prediction.')

        file_name_list = []
        preds_array = np.empty((0, size*size), dtype = np.long)

        data = dataset()
        #print(data)

        #for step, ids in enumerate(test_dataflow):
        #    image, image_infos = dataset(ids)
        images, image_infos = zip(*data)
        #print(images[0])
        #images, image_infos = data
        pred_list = []
        for image in images:
            preds = predict(image = np.asarray(image), session = sess, size = (size, size))
            pred_list.append(preds)

        for n, (preds, image_infos) in enumerate(zip(pred_list, image_infos)):

            segments = np.argmax(preds, axis = -1)

            #temp_mask = []
            #for img, mask in zip(np.stack(image), segments):
            #    transformed = transform(image = img, mask = mask)
            #    mask = transformed['mask']
            #    temp_mask.append(mask)

            #mask = np.array(temp_mask)

            mask = segments
            mask = mask.reshape([mask.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, mask))

            file_name_list.append([i['file_name'] for i in image_infos])

            save_image = (images[n][0] * 255).astype(np.uint8)
            save_segment = (segments[0] * 255 / 11).astype(np.uint8)

            name = os.path.basename(image_infos[0]['file_name']).split('.')[0]
            cv2.imwrite(pj(pred_dir, name + '_img.png'), save_image)
            cv2.imwrite(pj(pred_dir, name + '_seg.png'), save_segment[Ellipsis, None])

        print('End prediction.')
        file_names = [y for x in file_name_list for y in x]

        submission = pd.DataFrame()

        for file_name, string in zip(file_names, preds_array):
            submission = submission._append(
                    {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                    ignore_index = True
                    )

        submission.to_csv(pj(pred_dir, "junwoo_choi_segmentation.csv"), index = False)

