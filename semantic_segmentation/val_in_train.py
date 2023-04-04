import time
import shutil
import random
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


def evaluate(model, dataset_val, loader_val, config):
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    # build val dataset and dataloader
    # transforms_val = [Resize(target_size=config.VAL.IMAGE_BASE_SIZE,
    #                          keep_ori_size=config.VAL.KEEP_ORI_SIZE),  # KEEP_ORI_SIZE 默认为False，即不保保持原大小，需要改变大小
    #                   Normalize(mean=config.VAL.MEAN, std=config.VAL.STD)]
    #
    # dataset_val = get_dataset(config, data_transform=transforms_val, mode='val')
    # batch_sampler = paddle.io.DistributedBatchSampler(dataset_val,
    #                                                   batch_size=config.DATA.BATCH_SIZE_VAL, shuffle=True,
    #                                                   drop_last=True)
    # collate_fn = multi_val_fn()
    # loader_val = paddle.io.DataLoader(dataset_val, batch_sampler=batch_sampler,
    #                                   num_workers=config.DATA.NUM_WORKERS, return_list=True, collate_fn=collate_fn)

    total_iters = len(loader_val)

    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    logger.info("Start evaluating (total_samples: {}, total_iters: {})".format(len(dataset_val), total_iters))
    progbar_val = progbar.Progbar(target=total_iters, verbose=1)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    val_start_time = time.time()
    with paddle.no_grad():
        for iter, (img, label) in enumerate(loader_val):
            reader_cost_averager.record(time.time() - batch_start)
            batch_size = len(img)
            ori_shape = [l.shape[-2:] for l in label]

            pred = infer.ss_inference(
                model=model,
                img=img,
                ori_shape=ori_shape,  # 返回(1, num_classes, h, w), h,w就是ori.shape
                is_slide=True,  # 是否通过滑动窗口推断,默认为True
                base_size=config.VAL.IMAGE_BASE_SIZE,  # 当短边的尺寸小于min(base_size)时，将其调整为min(base_size)。
                stride_size=config.VAL.STRIDE_SIZE,  # 步长
                crop_size=config.VAL.CROP_SIZE,  # 滑动窗口的大小
                num_classes=config.DATA.NUM_CLASSES,
                rescale_from_ori=config.VAL.RESCALE_FROM_ORI)  # 是否从原始尺寸重新调整图像的大小。默认为False

            for i in range(batch_size):
                intersect_area, pred_area, label_area = metrics.calculate_area(
                    pred[i],
                    label[i],
                    dataset_val.num_classes,
                    ignore_index=dataset_val.ignore_index)

                # Gather from all ranks
                if nranks > 1:
                    intersect_area_list = []
                    pred_area_list = []
                    label_area_list = []
                    paddle.distributed.all_gather(intersect_area_list, intersect_area)
                    paddle.distributed.all_gather(pred_area_list, pred_area)
                    paddle.distributed.all_gather(label_area_list, label_area)
                    # Some image has been evaluated and should be eliminated in last iter
                    if (iter + 1) * nranks > len(dataset_val):
                        valid = len(dataset_val) - iter * nranks
                        intersect_area_list = intersect_area_list[:valid]
                        pred_area_list = pred_area_list[:valid]
                        label_area_list = label_area_list[:valid]
                    for i in range(len(intersect_area_list)):
                        intersect_area_all = intersect_area_all + intersect_area_list[i]
                        pred_area_all = pred_area_all + pred_area_list[i]
                        label_area_all = label_area_all + label_area_list[i]

                else:
                    intersect_area_all = intersect_area_all + intersect_area
                    pred_area_all = pred_area_all + pred_area
                    label_area_all = label_area_all + label_area

            batch_cost_averager.record(time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()
            if local_rank == 0:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost), ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()
    val_end_time = time.time()
    val_time_cost = val_end_time - val_start_time

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all, label_area_all)
    acc, class_precision, class_recall = metrics.accuracy(intersect_area_all, pred_area_all, label_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall)
    mf1 = np.nanmean(class_f1)
    # logger.info("Val_time_cost:   {}".format(val_time_cost))
    # logger.info("[EVAL] Images: {}  mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(len(dataset_val), miou, acc, kappa))
    # logger.info("[EVAL] Class IoU: " + str(np.round(class_iou, 4)))
    # logger.info("[EVAL] Class Acc: " + str(np.round(class_acc, 4)))

    return val_time_cost, miou, acc, kappa, class_iou, class_precision, class_f1, mf1
