import os
import time
import random
import argparse
import numpy as np
from collections import deque
import paddle
import paddle.nn as nn
from config import *
from src.utils import logger
from src.datasets import get_dataset
from src.models import get_model
from src.transforms import *
from src.utils import TimeAverager, calculate_eta, resume, get_dataloader, op_flops_funs
from src.models.solver import get_scheduler, get_optimizer
from src.models.losses import get_loss_function
from val_in_train import evaluate
from src.utils import multi_val_fn

# How to start training?
# train on single-gpu
# CUDA_VISIBLE_DEVICES=0 python3 train.py --config ./configs/EMRT/EMRT_256x256_160k_potsdam.yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual Transformer for semantic segmentation')
    parser.add_argument(
        "--config",
        dest='cfg',
        # default="configs/trans2seg/Trans2Seg_medium_256x256_160k_potsdam.yaml",

        # different backbone
        # default="configs/EMRT/EMRT_256x256_160k_potsdam.yaml",
        # default="configs/EMRT/EMRT_256x256_160k_potsdam_hrnet.yaml",
        # default="configs/EMRT/EMRT_256x256_160k_potsdam_cswin.yaml",
        # default="configs/EMRT/EMRT_256x256_160k_potsdam_vit.yaml",
        # default="configs/EMRT/EMRT_256x256_160k_potsdam_segformer.yaml",

        # different dataset
        default="configs/EMRT/EMRT_256x256_160k_potsdam.yaml",
        # default="configs/EMRT/EMRT_256x256_120k_vaihingen.yaml",
        # default="configs/EMRT/EMRT_256x256_160k_loveda.yaml",

        type=str,
        help="The config file."
    )
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=1234,
        type=int)
    return parser.parse_args()


def main():
    plot_corve = False
    config = get_config()
    args = parse_args()
    config = update_config(config, args)
    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    place = 'gpu' if config.TRAIN.USE_GPU else 'cpu'
    paddle.set_device(place)
    # build  model
    model = get_model(config)
    model.train()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank

    # build scheduler
    lr_scheduler = get_scheduler(config)
    # build optimizer
    optimizer = get_optimizer(model, lr_scheduler, config)
    # bulid train transforms
    transforms_train = get_transforms(config)
    # build dataset_train
    dataset_train = get_dataset(config, data_transform=transforms_train, mode='train')
    train_loader = get_dataloader(dataset=dataset_train,
                                  shuffle=True,
                                  batch_size=config.DATA.BATCH_SIZE,
                                  num_iters=config.TRAIN.ITERS,
                                  num_workers=config.DATA.NUM_WORKERS)

    # build val dataset and dataloader
    transforms_val = [Resize(target_size=config.VAL.IMAGE_BASE_SIZE,
                             keep_ori_size=config.VAL.KEEP_ORI_SIZE),
                      Normalize(mean=config.VAL.MEAN, std=config.VAL.STD)]

    dataset_val = get_dataset(config, data_transform=transforms_val, mode='val')
    batch_sampler = paddle.io.DistributedBatchSampler(dataset_val,
                                                      batch_size=config.DATA.BATCH_SIZE_VAL, shuffle=True,
                                                      drop_last=True)
    collate_fn = multi_val_fn()
    loader_val = paddle.io.DataLoader(dataset_val, batch_sampler=batch_sampler,
                                      num_workers=config.DATA.NUM_WORKERS, return_list=True, collate_fn=collate_fn)

    # build loss function
    loss_func = get_loss_function(config)
    # TODO(wutianyiRosun@gmail.com): Resume from checkpoints, and update start_iter

    # build workspace for saving checkpoints
    if not os.path.isdir(config.SAVE_DIR):
        if os.path.exists(config.SAVE_DIR):
            os.remove(config.SAVE_DIR)
        os.makedirs(config.SAVE_DIR)
    logger.info("train_cfg: {}".format(args.cfg))
    logger.info("train_model_name: {}".format(config.MODEL.NAME))
    logger.info("train_datatset: {}".format(config.DATA.DATASET))
    logger.info("train_loader.len: {}".format(len(train_loader)))

    start_iter = 0
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
            logger.info("using dist training")
            paddle.distributed.init_parallel_env()
            ddp_model = paddle.DataParallel(model)
        else:
            ddp_model = paddle.DataParallel(model)

    avg_loss = 0.0
    avg_loss_list = []
    best_mean_iou = -1.0
    best_acc = -1.0
    best_model_iter = -1
    iters_per_epoch = len(dataset_train) // config.DATA.BATCH_SIZE
    total_epoch = config.TRAIN.ITERS // iters_per_epoch
    logger.info("iters_per_epoch= {}".format(iters_per_epoch))

    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    save_models = deque()
    batch_start = time.time()
    cur_iter = start_iter

    # begin training
    for data in train_loader:
        cur_iter += 1
        reader_cost_averager.record(time.time() - batch_start)
        images = data[0]
        labels = data[1].astype('int64')
        if nranks > 1:
            logits_list = ddp_model(images)
        else:
            logits_list = model(images)

        loss_list = loss_func(logits_list, labels)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()

        lr = optimizer.get_lr()
        if isinstance(optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()
        model.clear_gradients()
        avg_loss += loss.numpy()[0]
        if not avg_loss_list:
            avg_loss_list = [l.numpy() for l in loss_list]
        else:
            for i in range(len(loss_list)):
                avg_loss_list[i] += loss_list[i].numpy()
        batch_cost_averager.record(
            time.time() - batch_start, num_samples=config.DATA.BATCH_SIZE)

        # print training log
        if (cur_iter) % config.LOGGING_INFO_FREQ == 0 and local_rank == 0:
            avg_loss /= config.LOGGING_INFO_FREQ
            avg_loss_list = [l[0] / config.LOGGING_INFO_FREQ for l in avg_loss_list]
            remain_iters = config.TRAIN.ITERS - cur_iter
            avg_train_batch_cost = batch_cost_averager.get_average()
            avg_train_reader_cost = reader_cost_averager.get_average()
            eta = calculate_eta(remain_iters, avg_train_batch_cost)
            logger.info(
                "[TRAIN] Epochs: {}/{}, iter: {}/{}, loss: {:.4f}, lr: {:.8f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}".format(
                    (cur_iter - 1) // iters_per_epoch + 1, total_epoch + 1, cur_iter, config.TRAIN.ITERS, avg_loss,
                    lr, avg_train_batch_cost, avg_train_reader_cost,
                    batch_cost_averager.get_ips_average(), eta))
            avg_loss = 0.0
            avg_loss_list = []
            reader_cost_averager.reset()
            batch_cost_averager.reset()

        if (cur_iter % config.SAVE_FREQ_CHECKPOINT == 0 or cur_iter == config.TRAIN.ITERS):
            val_time_cost, mean_iou, acc, kappa, class_iou, class_acc, class_f1, mean_f1 = evaluate(model, dataset_val,
                                                                                                loader_val, config)
            logger.info("Val_time_cost:   {}".format(val_time_cost))
            logger.info("In this val: mIoU {:.4f},  Acc: {:.4f}, F1-Score:{:.4f}".format(mean_iou, acc, mean_f1))
            logger.info(
                "Current best_mIoU: {:.4f},  Acc: {:.4f}, iter: {}".format(best_mean_iou, best_acc, best_model_iter))

            model.train()

        if (cur_iter % config.SAVE_FREQ_CHECKPOINT == 0 or cur_iter == config.TRAIN.ITERS) and local_rank == 0:
            current_save_weigth_file = os.path.join(config.SAVE_DIR,
                                                    "iter_{}_model_state.pdparams".format(cur_iter))
            current_save_opt_file = os.path.join(config.SAVE_DIR,
                                                 "iter_{}_opt_state.pdopt".format(cur_iter))

            paddle.save(model.state_dict(), current_save_weigth_file)
            paddle.save(optimizer.state_dict(), current_save_opt_file)
            save_models.append([current_save_weigth_file,
                                current_save_opt_file])
            logger.info("saving the weights of model to {}".format(
                current_save_weigth_file))

            if len(save_models) > config.KEEP_CHECKPOINT_MAX > 0:
                files_to_remove = save_models.popleft()
                os.remove(files_to_remove[0])
                os.remove(files_to_remove[1])

            # 保存最佳模型
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                best_acc = acc
                best_model_iter = cur_iter
                best_model_dir = os.path.join(config.SAVE_DIR, "best_model.pdparams")
                paddle.save(model.state_dict(), best_model_dir)

                logger.info('\n[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'.format(
                    best_mean_iou, best_model_iter))
                logger.info(
                    "[EVAL] Images: {}  mIoU: {:.4f}  Acc: {:.4f}  Kappa: {:.4f}  mean_f1: {:.4f}".format(len(dataset_val), mean_iou,
                                                                                          acc, kappa, mean_f1))
                logger.info("[EVAL] Class IoU: " + str(np.round(class_iou, 4)))
                logger.info("[EVAL] Class Acc: " + str(np.round(class_acc, 4)))
                logger.info("[EVAL] Class F1-score: " + str(np.round(class_f1, 4)) + "\n")

        batch_start = time.time()

    print(
        "\ntrain_model_name: {}, train_dataset: {}, train_loader.len: {}".format(config.MODEL.NAME, config.DATA.DATASET,
                                                                                 len(train_loader)))
    print('[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.\n'.format(best_mean_iou,
                                                                                                   best_model_iter))

    # Calculate flops.
    if local_rank == 0:
        _, c, h, w = images.shape
        _ = paddle.flops(
            model, [1, c, h, w], custom_ops={paddle.nn.SyncBatchNorm: op_flops_funs.count_syncbn})

    # Calculate parameter
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for p in model.parameters():
        mulValue = np.prod(p.shape)
        Total_params += mulValue
        if p.stop_gradient:
            NonTrainable_params += mulValue
        else:
            Trainable_params += mulValue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

    # Sleep for half a second to let dataloader release resources.
    time.sleep(1.0)

if __name__ == '__main__':
    main()
