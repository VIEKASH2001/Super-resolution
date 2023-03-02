#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
import os
from time import time

import torch

import utils.network_utils as net_utils
from losses.metrics import *


def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    os.mkdir(path)


def val(cfg, epoch_idx, val_data_loader, net, val_writer):
    Init_Epoch = epoch_idx

    n_batches = len(val_data_loader)

    # Batch average meterics
    batch_time = net_utils.AverageMeter()
    data_time = net_utils.AverageMeter()
    PSNRs = net_utils.AverageMeter()

    batch_end_time = time()

    for batch_idx, (_, imgs_lr_s, imgs_lr, imgs_hr) in enumerate(val_data_loader):
        data_time.update(time() - batch_end_time)
        # Switch models to validation mode
        net.eval()

        with torch.no_grad():
            # Get data from data loader
            # imgs_lr_s, imgs_lr, imgs_hr = [
            #     [net_utils.var_or_cuda(img) for img in stack] for stack in [imgs_lr_s, imgs_lr, imgs_hr]
            # ]
            imgs_lr_s, imgs_lr = [[net_utils.var_or_cuda(img) for img in stack] for stack in [imgs_lr_s, imgs_lr]]

            if cfg.TEST.CHOP:
                img_out = net_utils.Chop(model=net)(imgs_lr_s, imgs_lr)
            else:
                img_out = net(imgs_lr_s, imgs_lr)

            if batch_idx < cfg.VAL.VISUALIZATION_NUM:
                if epoch_idx == Init_Epoch:  # This condition actually does not do anything. Always is True.
                    img_lr_cpu = imgs_lr[cfg.NETWORK.TEMPORAL_WIDTH // 2][0].cpu() / 255.0  # [0] -> 1st of the batch
                    img_hr_cpu = imgs_hr[cfg.NETWORK.TEMPORAL_WIDTH // 2][0].cpu() / 255.0  # dont use squeeze (bs>1)
                    val_writer.add_image(cfg.CONST.NAME + "/IMG_LR" + str(batch_idx + 1), img_lr_cpu, epoch_idx + 1)
                    val_writer.add_image(cfg.CONST.NAME + "/IMG_HR" + str(batch_idx + 1), img_hr_cpu, epoch_idx + 1)

                img_out_cpu = img_out[0].cpu().clamp(0, cfg.DATA.RANGE) / 255.0
                val_writer.add_image(cfg.CONST.NAME + "/IMG_OUT" + str(batch_idx + 1), img_out_cpu, epoch_idx + 1)

            img_out = net_utils.tensor2img(img_out, min_max=[0, cfg.DATA.RANGE])
            imgs_hr = net_utils.tensor2img(imgs_hr[cfg.NETWORK.TEMPORAL_WIDTH // 2], min_max=[0, cfg.DATA.RANGE])

            img_out, imgs_hr = net_utils.crop_border([img_out, imgs_hr], cfg.CONST.SCALE)

            img_out_y = net_utils.bgr2ycbcr(img_out / 255.0, only_y=True)
            imgs_hr_y = net_utils.bgr2ycbcr(imgs_hr / 255.0, only_y=True)

            PSNR = calculate_psnr(img_out_y * 255, imgs_hr_y * 255)
            PSNRs.update(PSNR, cfg.CONST.VAL_BATCH_SIZE)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.VAL.PRINT_FREQ == 0:
                print(
                    "[VAL] [Epoch {0}/{1}][Batch {2}/{3}]\t BT {4}\t DT {5}\t PSNR {6}".format(
                        epoch_idx + 1, cfg.TRAIN.NUM_EPOCHS, batch_idx + 1, n_batches, batch_time, data_time, PSNRs
                    )
                )

    # Add validation results to TensorBoard
    val_writer.add_scalar(cfg.CONST.NAME + "/PSNR_VAL", PSNRs.avg, epoch_idx + 1)

    print("============================ RESULTS ===========================")
    print("[VAL] Average_PSNR: " + str(PSNRs.avg))
    print(
        "[VAL] [Epoch {0}] BatchTime_avg {1} DataTime_avg {2} PSNR_avg {3}\n".format(
            epoch_idx + 1, batch_time.avg, data_time.avg, PSNRs.avg
        )
    )
    return PSNRs.avg
