#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import os

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Common
#
__C.CONST = edict()
__C.CONST.NUM_GPU = 2
__C.CONST.NUM_WORKER = 4  # number of data workers
__C.CONST.TRAIN_BATCH_SIZE = 2 if os.uname().nodename == "mmlab" else 1
__C.CONST.VAL_BATCH_SIZE = 1
__C.CONST.TEST_BATCH_SIZE = 1
__C.CONST.NAME = "IGNNSR"
__C.CONST.WANDB = True

__C.CONST.WEIGHTS = "./ckpt/IGNN_x4.pth"
__C.CONST.SCALE = 4

#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.DATASET_TRAIN_NAME = "RatColon" if os.uname().nodename == "mmlab" else "TestRatColon"  # DIV2K
# DIV2K_val, Set5, Set14, BSD100, Urban100, Manga109, Demo
__C.DATASET.DATASET_TEST_NAME = "RatColon" if os.uname().nodename == "mmlab" else "TestRatColon"

#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = "./output"
__C.DIR.DATASET_SCALE = "x" + str(__C.CONST.SCALE)

# For DIV2K
__C.DIR.DATASET_ROOT = "datasets/RatColon/" if os.uname().nodename == "mmlab" else "datasets/TestRatColon/"
# if cfg.DATASET.DATASET_TRAIN_NAME == 'DIV2K':
#     __C.DIR.DATASET_JSON_TRAIN_PATH     = './datasets/json_files/DIV2K.json'
#     __C.DIR.IMAGE_LR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_LR_bicubic_sub/'+__C.DIR.DATASET_SCALE+'/{1}.png'
#     __C.DIR.IMAGE_HR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_HR_sub/{1}.png'
#     # __C.DIR.IMAGE_HR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_HR_sub/'+__C.DIR.DATASET_SCALE+'/{1}.png'

if cfg.DATASET.DATASET_TRAIN_NAME == ("RatColon" if os.uname().nodename == "mmlab" else "TestRatColon"):
    __C.DIR.DATASET_JSON_TRAIN_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/train.json"
    __C.DIR.IMAGE_LR_TRAIN_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/{0}_LR_sub/" + __C.DIR.DATASET_SCALE + "/{1}.png"
    __C.DIR.IMAGE_HR_TRAIN_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/{0}_HR_sub/{1}.png"
    # __C.DIR.IMAGE_HR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_HR_sub/'+__C.DIR.DATASET_SCALE+'/{1}.png'

# For DIV2K validation
# if cfg.DATASET.DATASET_TEST_NAME == 'DIV2K_val':
#     __C.DIR.DATASET_JSON_TEST_PATH      = './datasets/json_files/DIV2K_val.json'
#     __C.DIR.IMAGE_LR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_LR_bicubic/'+__C.DIR.DATASET_SCALE+'/{1}.png'
#     __C.DIR.IMAGE_HR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'DIV2K/DIV2K_{0}_HR/{1}.png'

if cfg.DATASET.DATASET_TEST_NAME == ("RatColon" if os.uname().nodename == "mmlab" else "TestRatColon"):
    __C.DIR.DATASET_JSON_TEST_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/valid.json"
    __C.DIR.IMAGE_LR_TEST_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/{0}_LR/" + __C.DIR.DATASET_SCALE + "/{1}.png"
    __C.DIR.IMAGE_HR_TEST_PATH = __C.DIR.DATASET_ROOT + "SR_Dataset/{0}_HR/{1}.png"

# For Set5, Set14, BSD100, Urban100, Manga109
if cfg.DATASET.DATASET_TEST_NAME in ["Set5", "Set14", "BSD100", "Urban100", "Manga109"]:
    __C.DIR.DATASET_JSON_TEST_PATH = "./datasets/json_files/" + __C.DATASET.DATASET_TEST_NAME + ".json"
    __C.DIR.IMAGE_LR_TEST_PATH = (
        __C.DIR.DATASET_ROOT + __C.DATASET.DATASET_TEST_NAME + "/LR/" + __C.DIR.DATASET_SCALE + "/{1}.png"
    )
    __C.DIR.IMAGE_HR_TEST_PATH = (
        __C.DIR.DATASET_ROOT + __C.DATASET.DATASET_TEST_NAME + "/HR/" + __C.DIR.DATASET_SCALE + "/{1}.png"
    )

# For Test images
elif cfg.DATASET.DATASET_TEST_NAME == "Demo":
    __C.DIR.IMAGE_LR_TEST_PATH = "./datasets/demo_test_images"

#
# data augmentation
#
__C.DATA = edict()
__C.DATA.RANGE = 255
__C.DATA.STD = [255.0, 255.0, 255.0]
__C.DATA.MEAN = [0.0, 0.0, 0.0]
__C.DATA.GAUSSIAN = 9  # RandomGaussianNoise
__C.DATA.COLOR_JITTER = [0.2, 0.15, 0.3, 0.1]  # brightness, contrast, saturation, hue

if cfg.CONST.SCALE == 2:
    __C.DATA.CROP_IMG_SIZE = [160, 160]
if cfg.CONST.SCALE == 3:
    __C.DATA.CROP_IMG_SIZE = [198, 198]
if cfg.CONST.SCALE == 4:
    __C.DATA.CROP_IMG_SIZE = [160, 160]

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.SRNETARCH = "IGNN"  # available options: IGNN
__C.NETWORK.LEAKY_VALUE = 0.0  # when value = 0.0, lrelu->relu
__C.NETWORK.RES_SCALE = 0.1  # 0.1 for edsr, 1 for baseline edsr
__C.NETWORK.N_RESBLOCK = 32
__C.NETWORK.N_FEATURE = 256
__C.NETWORK.N_NEIGHBOR = 5
__C.NETWORK.WITH_WINDOW = True
__C.NETWORK.WINDOW_SIZE = 30
__C.NETWORK.WITH_ADAIN_NORM = True
__C.NETWORK.WITH_DIFF = True
__C.NETWORK.WITH_SCORE = False
# Modifications
__C.NETWORK.TEMPORAL_WIDTH = 5
__C.NETWORK.OWN_SON = False  # If to consider I_{L_s}^{t} as well into search space
__C.NETWORK.WIDE_TEMPORAL = False  # Refer to architectures for info TODO: Implement wide temporal
__C.NETWORK.TOPK_SEARCH_SPACE = "individual"  # options: "combined" | "individual" TODO: Implement combined

__C.NETWORK.PHASE = "train"  # available options: 'train', 'test', 'resume'

#
# Training
#

__C.TRAIN = edict()
__C.TRAIN.PIXEL_LOSS = "L1"  # available options: 'L1', 'MSE'
__C.TRAIN.USE_PERCET_LOSS = False
__C.TRAIN.NUM_EPOCHS = 40 * __C.CONST.TRAIN_BATCH_SIZE  # maximum number of epochs, bs_2:80, bs_4:160 bs_8:320
__C.TRAIN.MAX_INTERS_PER_EPOCH = 30 if os.uname().nodename == "mmlab" else 40
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.LR_MILESTONES = [t * __C.CONST.TRAIN_BATCH_SIZE for t in [8, 16, 24, 32]]
__C.TRAIN.LR_DECAY = 0.5  # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BETA = 0.999
__C.TRAIN.BIAS_DECAY = 0.0  # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY = 0.0  # regularization of weight, default: 0
__C.TRAIN.KAIMING_SCALE = 0.1
__C.TRAIN.PRINT_FREQ = 10
__C.TRAIN.SAVE_FREQ = 5  # weights will be overwritten every save_freq epoch
__C.TRAIN.TEST_FREQ = 1

#
# Validating options
#
__C.VAL = edict()
__C.VAL.VISUALIZATION_NUM = 4
__C.VAL.PRINT_FREQ = 5

#
# Testing options
#
__C.TEST = edict()
__C.TEST.RUNTIME = False
__C.TEST.SAVEIMG = True
__C.TEST.CHOP = True
__C.TEST.ENSEMBLE = False
