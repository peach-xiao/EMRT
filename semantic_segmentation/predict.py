#  Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import shutil
import random
import math
import cv2
import argparse
import numpy as np
import paddle
import paddle.nn.functional as F
from config import *
from src.api import infer
from src.datasets import get_dataset
from src.transforms import Resize, Normalize
from src.models import get_model
from src.utils import multi_val_fn
from src.utils import metrics, logger, progbar
from src.utils import TimeAverager, calculate_eta
from src.utils import load_entire_model, resume


# CUDA_VISIBLE_DEVICES=3 python3  predict.py
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Seg. Models')
    parser.add_argument(
        "--config",
        dest='cfg',

        # default="configs/deformable_trans/EMRT_256x256_160k_potsdam.yaml",
        # default="configs/deformable_trans/EMRT_256x256_120k_vaihingen.yaml",
        # default="configs/deformable_trans/EMRT_256x256_160k_loveda.yaml",
        type=str,
        help='The config file.'
    )
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of weights file (segmentation model)',
        type=str,

        # default="/data/sdu02_peach/Paddle/EMRT_256x256_120k_vaihingen_pretrain/best_model.pdparams"
        # default ="/data/sdu02_peach/Paddle/EMRT_256x256_160k_potsdam_pretrain/best_model.pdparams"
        # default = "/data/sdu02_peach/Paddle/EMRT_256x256_160k_loveda_pretrain/best_model.pdparams"

    )

    parser.add_argument(
        "--multi_scales",
        type=bool,
        default=False,
        help='whether employing multiple scales testing'
    )
    return parser.parse_args()


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def get_palette(dataset_name):
    if dataset_name.lower() == "potsdam" or dataset_name.lower() == "vaihingen":
        label_values = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter/background']
        palette = {0: (255, 255, 255),  # 'Impervious surfaces'
                   1: (0, 0, 255),  # 'Building'
                   2: (0, 255, 255),  # 'Low vegetation'
                   3: (0, 255, 0),  # 'Tree'
                   4: (255, 255, 0),  # 'Car'
                   5: (255, 0, 0)}  # 'Clutter/background'

    elif dataset_name.lower() == "loveda":
        label_values = ['imp_surfaces', 'building', 'low_vegetation',
                        'tree', 'car', 'clutter']

        palette = {0: (255, 255, 255),  # Background
                   1: (255, 0, 0),  # Building
                   2: (255, 255, 0),  # Road
                   3: (0, 0, 255),  # Water
                   4: (159, 129, 183),  # Barren
                   5: (0, 255, 0),  # Forest
                   6: (255, 195, 128)}  # Agricultural
    return palette

if __name__ == '__main__':

    config = get_config()
    args = parse_args()
    config = update_config(config, args)
    pred_saved_dir = "//data//sdu02_peach//torch_vit_pth//predict_paddle_resnet50//"+config.DATA.DATASET.lower()+"//"+config.MODEL.NAME.lower()
    logger.info("pred_saved_dir: {}".format(pred_saved_dir))
    if args.model_path is None:
        args.model_path = os.path.join(config.SAVE_DIR,
                                       "iter_{}_model_state.pdparams".format(config.TRAIN.ITERS))
    place = 'gpu' if config.VAL.USE_GPU else 'cpu'
    paddle.set_device(place)
    # build model
    model = get_model(config)
    if args.model_path:
        load_entire_model(model, args.model_path)
        logger.info('Loaded trained params of model successfully')
    model.eval()

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    palette = get_palette(config.DATA.DATASET)

    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    # build val dataset and dataloader
    transforms_val = [Resize(target_size=config.VAL.IMAGE_BASE_SIZE,
                             keep_ori_size=config.VAL.KEEP_ORI_SIZE),
                      Normalize(mean=config.VAL.MEAN, std=config.VAL.STD)]

    dataset_val = get_dataset(config, data_transform=transforms_val, mode = 'test')
    batch_sampler = paddle.io.DistributedBatchSampler(dataset_val, batch_size=1, shuffle=False, drop_last=False)

    collate_fn = multi_val_fn()
    loader_val = paddle.io.DataLoader(dataset_val, batch_sampler=batch_sampler,
                                      num_workers=config.DATA.NUM_WORKERS, return_list=True, collate_fn=collate_fn)

    total_iters = len(loader_val)

    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    logger.info("Start predicting (total_samples: {}, total_iters: {}, "
                "multi-scale testing: {})".format(len(dataset_val), total_iters, args.multi_scales))

    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    with paddle.no_grad():
        for iter, (img, label) in enumerate(loader_val):
            reader_cost_averager.record(time.time() - batch_start)

            img = img[0]
            img = img[np.newaxis, ...]
            img = paddle.to_tensor(img)
            logit = model(img)[0]
            logit = F.softmax(logit, axis=1)
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')

            if not os.path.exists(pred_saved_dir):
                os.makedirs(pred_saved_dir)

            color_label = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for c, i in palette.items():
                m = pred == c
                color_label[m] = i
            color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)
            cv2.imwrite(pred_saved_dir + "//" + str(iter) + '.png', color_label)


            batch_cost_averager.record(time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()
            if local_rank == 0:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost), ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    logger.info("Predict Done!")


def predict(dataloader, model, total_batch, debug_steps=100, logger=None):
    """Predict for whole dataset
    Args:
        dataloader: paddle.io.DataLoader, dataloader instance
        model: nn.Layer, a ViT model
        total_batch: int, total num of batches for one epoch
        debug_steps: int, num of iters to log info, default: 100
        logger: logger for logging, default: None
    Returns:
        preds: prediction results
        pred_time: float, prediction time
    """
    model.eval()
    time_st = time.time()

    preds = []
    with paddle.no_grad():
        for batch_id, data in enumerate(dataloader):
            image = data[0]
            output = model(image)
            pred = F.softmax(output)
            preds.append(pred)
            if logger and batch_id % debug_steps == 0:
                logger.info(f"Pred Step[{batch_id:04d}/{total_batch:04d}], done")

    pred_time = time.time() - time_st
    return preds, pred_time
