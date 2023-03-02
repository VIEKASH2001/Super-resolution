#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

import io
import json
import os
from datetime import datetime as dt
from enum import Enum

import cv2
import torch.utils.data.dataset

from config import cfg
from utils.imresize import imresize
from utils.network_utils import img2tensor


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1


class SRDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms
        if cfg.CONST.SCALE == 4:
            self.down_scale = 2
        else:
            self.down_scale = cfg.CONST.SCALE

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name, imgs_lr, imgs_hr = self.image_read(idx)
        imgs_lr, imgs_hr = self.transforms(imgs_lr, imgs_hr)
        imgs_lr_s = []
        for each_img_lr in imgs_lr:
            _, h, w = each_img_lr.size()
            img_lr_ = each_img_lr[:, : int(h - h % self.down_scale), : int(w - w % self.down_scale)]
            img_lr_s = imresize(img_lr_ / 255.0, 1.0 / self.down_scale) * 255
            img_lr_s = img_lr_s.clamp(0, 255)
            imgs_lr_s.append(img_lr_s)
        return img_name, imgs_lr_s, imgs_lr, imgs_hr

    def image_read(self, idx):
        return (
            self.file_list[idx]["img_name"],
            [cv2.imread(self.file_list[idx]["imgs_lr"][i]) for i in range(len(self.file_list[idx]["imgs_lr"]))],
            [cv2.imread(self.file_list[idx]["imgs_hr"][i]) for i in range(len(self.file_list[idx]["imgs_hr"]))],
        )


# //////////////////////////////// = End of SRDataset Class Definition = ///////////////////////////////// #


class SRDataLoader:
    def __init__(self, dataset_type, temporal_width):
        self.dataset_type = dataset_type
        self.temporal_width = list(range(-temporal_width // 2 + 1, temporal_width // 2 + 1, 1))
        if dataset_type == DatasetType.TRAIN:  # for train dataset
            self.img_lr_path_template = cfg.DIR.IMAGE_LR_TRAIN_PATH
            self.img_hr_path_template = cfg.DIR.IMAGE_HR_TRAIN_PATH
            with io.open(cfg.DIR.DATASET_JSON_TRAIN_PATH, encoding="utf-8") as file:
                self.files_list = json.loads(file.read())
        elif dataset_type == DatasetType.TEST:  # for val/test dataset
            self.img_lr_path_template = cfg.DIR.IMAGE_LR_TEST_PATH
            self.img_hr_path_template = cfg.DIR.IMAGE_HR_TEST_PATH
            with io.open(cfg.DIR.DATASET_JSON_TEST_PATH, encoding="utf-8") as file:
                self.files_list = json.loads(file.read())

    def get_dataset(self, transforms=None):
        files = []
        # Load data for each category
        for file in self.files_list:
            if self.dataset_type == DatasetType.TRAIN and file["phase"] == "train":
                phase = file["phase"]
                samples = file["sample"]
                print("[INFO] %s Collecting files [phase = %s]" % (dt.now(), phase))
                files.extend(self.get_files(phase, samples))
            elif self.dataset_type == DatasetType.TEST and file["phase"] in ["valid", "test"]:
                phase = file["phase"]
                samples = file["sample"]
                print("[INFO] %s Collecting files [phase = %s]" % (dt.now(), phase))
                files.extend(self.get_files(phase, samples))

        print(
            "[INFO] %s Complete collecting files of the dataset for %s. Total images: %d."
            % (dt.now(), self.dataset_type.name, len(files))
        )
        return SRDataset(files, transforms)

    def get_files(self, phase, samples):
        files = []
        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            break_ = False
            datapoint_lr = []  # Stacked Images
            datapoint_hr = []  # Stacked Images
            for offset in self.temporal_width:
                offset_sample_name = int(sample_name.split("_")[0]) + offset
                if phase == "train":
                    offset_sample_name = str(offset_sample_name) + "_" + "".join(sample_name.split("_")[1:])
                else:
                    offset_sample_name = str(offset_sample_name)
                img_lr_path = self.img_lr_path_template.format(phase, offset_sample_name)
                img_hr_path = self.img_hr_path_template.format(phase, offset_sample_name)
                if os.path.exists(img_lr_path) and os.path.exists(img_hr_path):
                    datapoint_lr.append(img_lr_path)
                    datapoint_hr.append(img_hr_path)
                else:
                    # Points without a sufficiently large neighbourhood have to be skipped (break then continue)
                    break_ = True
                    break
            if break_:
                continue
            files.append(
                {
                    "img_name": sample_name,
                    "imgs_lr": datapoint_lr,
                    "imgs_hr": datapoint_hr,
                }
            )
        return files


# /////////////////////////////// = End of SRDataLoader Class Definition = /////////////////////////////// #


class TestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        # self.transforms = transforms
        if cfg.CONST.SCALE == 4:
            self.down_scale = 2
        else:
            self.down_scale = cfg.CONST.SCALE

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name, img_lr = self.image_read(idx)
        img_lr = img_lr[:, :, [2, 1, 0]]
        img_lr = img2tensor(img_lr)
        _, h, w = img_lr.size()
        img_lr_ = img_lr[:, : int(h - h % self.down_scale), : int(w - w % self.down_scale)]
        img_lr_s = imresize(img_lr_ / 255.0, 1.0 / self.down_scale) * 255

        img_lr_s = img_lr_s.clamp(0, 255)
        return img_name, img_lr_s, img_lr

    def image_read(self, idx):
        return self.file_list[idx]["img_name"], cv2.imread(self.file_list[idx]["img_lr"])


# //////////////////////////////// = End of TestDataset Class Definition = ///////////////////////////////// #


class TestDataLoader:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.img_lr_path = cfg.DIR.IMAGE_LR_TEST_PATH
        # Load all files of the dataset
        self.samples = sorted(os.listdir(self.img_lr_path))

    def get_dataset(self, transforms=None):
        assert self.dataset_type == DatasetType.TEST
        files = []
        # Load data for each category
        for sample_idx, sample_name in enumerate(self.samples):
            # Get file path of img
            img_lr_path = os.path.join(self.img_lr_path, sample_name)
            if os.path.exists(img_lr_path):
                files.append({"img_name": sample_name[:-4], "img_lr": img_lr_path})

        print(
            "[INFO] %s Complete collecting files for %s. Total test images: %d."
            % (dt.now(), self.dataset_type.name, len(files))
        )
        return TestDataset(files, transforms)


# /////////////////////////////// = End of TestDataLoader Class Definition = /////////////////////////////// #


# Datasets MAP
DATASET_LOADER_MAPPING = {
    "DIV2K": SRDataLoader,
    "DIV2K_val": SRDataLoader,
    "Set5": SRDataLoader,
    "Set14": SRDataLoader,
    "BSD100": SRDataLoader,
    "Urban100": SRDataLoader,
    "Manga109": SRDataLoader,
    "Demo": TestDataLoader,
    "RatColon": SRDataLoader,
    "TestRatColon": SRDataLoader,
}
