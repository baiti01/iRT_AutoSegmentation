#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/6/2019 3:12 PM

# sys
import os
import numpy as np
import SimpleITK as sitk
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch
import torch
import torch.utils.data as data

# monai
from monai.utils import MAX_SEED
from monai.transforms import (
    Randomizable,
    apply_transform,
    Compose,
    ResizeWithPadOrCropd,
    RandSpatialCropd,
    RandRotated,
    RandZoomd,
)


class IPADataset(data.Dataset, Randomizable):
    def __init__(self,
                 data_root=r'./data',
                 data_list=r'train.txt',
                 transforms=None,
                 customized_dataset_size=0
                 ):
        super(IPADataset, self).__init__()

        self.data_root = data_root
        self.transforms = transforms

        self.sample_list = []
        with open(os.path.join(self.data_root, data_list), 'r') as f:
            self.sample_list = [x.strip() for x in f.readlines()]

        self._seed = 0
        self.real_dataset_size = len(self.sample_list)
        self.dataset_size = customized_dataset_size if customized_dataset_size else self.real_dataset_size

    def randomize(self):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, item):
        self.randomize()
        item = item % self.real_dataset_size
        sample_idx = self.sample_list[item]

        sample_CT_image = sitk.ReadImage(os.path.join(self.data_root, sample_idx, 'CT.nii.gz'), imageIO=r'NiftiImageIO')
        sample_CT_image = sitk.GetArrayFromImage(sample_CT_image).astype(np.float32)
        sample_CT_image = torch.from_numpy(sample_CT_image).unsqueeze(0)
        sample_CT_image = (sample_CT_image + 1000) / 2000

        sample_MR_image = sitk.ReadImage(os.path.join(self.data_root, sample_idx, 'MRI.nii.gz'), imageIO=r'NiftiImageIO')
        sample_MR_image = sitk.GetArrayFromImage(sample_MR_image).astype(np.float32)
        sample_MR_image = torch.from_numpy(sample_MR_image).unsqueeze(0)
        sample_MR_image = sample_MR_image/ 2000

        sample_mask = sitk.ReadImage(os.path.join(self.data_root, sample_idx, 'mask.nii.gz'), imageIO=r'NiftiImageIO')
        sample_mask = sitk.GetArrayFromImage(sample_mask).astype(np.float32)
        sample_mask = torch.from_numpy(sample_mask).unsqueeze(0)

        result = apply_transform(self.transforms, data={'image': sample_CT_image, 'label': sample_mask})
        return {'input': result['image'],
                'target': result['label'],
                'current_data_path': sample_idx}

    def __len__(self):
        return self.dataset_size


def get_data_provider(cfg, phase='Train'):
    target_size = cfg.DATASET.TARGET_SIZE
    rand_crop_size = cfg.DATASET.RANDOM_CROP_SIZE
    rotation_probability = cfg.DATASET.ROTATION_RADIUS_PROBABILITY_ANGLE[0]
    rotation_angle_z, rotation_angle_y, rotation_angle_x = cfg.DATASET.ROTATION_RADIUS_PROBABILITY_ANGLE[1:]
    zoom_probability = cfg.DATASET.ZOOM_PROBABILITY_MIN_MAX_RATIO[0]
    zoom_min_ratio, zoom_max_ratio = cfg.DATASET.ZOOM_PROBABILITY_MIN_MAX_RATIO[1:]

    if phase.upper() == 'TRAIN':
        transform = Compose([
        RandSpatialCropd(keys=['image', 'label'],
                         roi_size=rand_crop_size,
                         random_size=False),
        RandRotated(keys=['image', 'label'],
                    range_x=rotation_angle_x,
                    range_y=rotation_angle_y,
                    range_z=rotation_angle_z,
                    prob=rotation_probability,
                    mode=['bilinear', 'nearest']),
        RandZoomd(keys=['image', 'label'],
                  mode=['trilinear', 'nearest'],
                  min_zoom=zoom_min_ratio,
                  max_zoom=zoom_max_ratio,
                  prob=zoom_probability),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=target_size),
        ])
    else:
        transform = Compose([
            ResizeWithPadOrCropd(keys=['image', 'label'],
                                 spatial_size=target_size)
        ])

    data_list = {'TRAIN': cfg.DATASET.TRAIN_LIST,
                 'VAL': cfg.DATASET.VAL_LIST,
                 'TEST': cfg.DATASET.TEST_LIST}

    batch_size = eval('cfg.{}.BATCHSIZE_PER_GPU'.format(phase.upper()))
    if torch.cuda.is_available():
        batch_size = batch_size * torch.cuda.device_count()

    iteration = int(cfg.TRAIN.TOTAL_ITERATION)
    current_dataset = IPADataset(data_root=cfg.DATASET.ROOT,
                                 data_list=data_list[phase.upper()],
                                 transforms = transform,
                                 customized_dataset_size=batch_size * iteration if phase.upper() == 'TRAIN' else 0
                                 )

    data_loader = torch.utils.data.DataLoader(current_dataset,
                                              batch_size=batch_size,
                                              shuffle=True if phase.upper() == 'Train' else False,
                                              num_workers=cfg.WORKERS,
                                              pin_memory=torch.cuda.is_available())

    return data_loader


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    random_crop_size = (160, 384, 384)
    target_size = (256, 512, 512)
    data_root = r'D:\data\0_curated_data\segmentation\3_Pelvis\IPA'
    data_list = 'IPA_train.txt'
    is_train = True
    is_shuffle = True

    batch_size = 1
    num_threads = 0
    is_gpu = torch.cuda.is_available()

    train_transform = Compose([
        RandSpatialCropd(keys=['image', 'label'],
                         roi_size=random_crop_size,
                         random_size=False),
        RandRotated(keys=['image', 'label'],
                    range_x=45/180,
                    range_y=2/180,
                    range_z=2/180,
                    prob=1,
                    mode=['bilinear', 'nearest']
                    ),
        RandZoomd(keys=['image', 'label'],
                  mode=['trilinear', 'nearest'],
                  min_zoom=0.8,
                  max_zoom=1.2,
                  prob=1),
        ResizeWithPadOrCropd(keys=['image', 'label'],
                             spatial_size=target_size),
    ])

    current_dataset = IPADataset(data_root=data_root,
                                 data_list=data_list,
                                 transforms=train_transform,
                                 customized_dataset_size=0)

    train_loader = torch.utils.data.DataLoader(current_dataset,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle,
                                               num_workers=num_threads,
                                               pin_memory=is_gpu)

    for i, data in enumerate(train_loader):
        current_image, current_target = data['input'], data['target']
        current_CT = current_image[:, 0].squeeze().detach().cpu().numpy()
        #current_MR = current_image[:, 1].squeeze().detach().cpu().numpy()
        current_target = current_target.squeeze().detach().cpu().numpy()

        idx = current_CT.shape[1] // 2
        current_CT = current_CT[128]
        #current_MR = current_MR[128]
        current_target = current_target[128]

        print("iter {}, "
              "shape: {}"
              "CT min/max: {}/{}, "
              "target min/max: {}/{}".format(i,
                                             current_CT.shape,
                                             np.min(current_CT), np.max(current_CT),
                                             #np.min(current_MR), np.max(current_MR),
                                             np.min(current_target), np.max(current_target)))
        if True:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(current_CT, cmap='gray', vmin=-250 / 2000 + 0.5, vmax=250 / 2000 + 0.5)
            plt.subplot(1, 3, 2)
            #plt.imshow(current_MR, cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(current_target, cmap='gray')
            plt.show()

    print('Congrats! May the force be with you ...')
