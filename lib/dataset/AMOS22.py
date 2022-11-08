#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/6/2019 3:12 PM

# sys
import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch
import torch
import torch.utils.data as data

# monai
from monai.utils import MAX_SEED
from monai.transforms import (
    AddChanneld,
    Orientationd,
    EnsureTyped,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByLabelClassesd,
    RandAffined,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd,
    Randomizable,
    apply_transform,
    Compose
)
from monai.data import partition_dataset, select_cross_validation_folds


class AMOS22Dataset(data.Dataset, Randomizable):
    def __init__(self,
                 data_root=r'./data',
                 data_list=None,
                 transforms=None,
                 customized_dataset_size=0
                 ):
        super(AMOS22Dataset, self).__init__()

        self.data_root = data_root
        self.transforms = transforms

        self.sample_list = data_list

        self._seed = 0
        self.real_dataset_size = len(self.sample_list)
        self.dataset_size = customized_dataset_size if customized_dataset_size else self.real_dataset_size

    def __randomize__(self):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, item):
        self.__randomize__()
        item = item % self.real_dataset_size
        image_path, label_path = self.sample_list[item]['image'], self.sample_list[item]['label']

        data = {}
        image = sitk.ReadImage(os.path.join(self.data_root, image_path), imageIO=r'NiftiImageIO')

        spacing = image.GetSpacing()
        direction = image.GetDirection()
        origin = image.GetOrigin()

        image = sitk.GetArrayFromImage(image).astype(np.float32)
        image = torch.from_numpy(image)
        data['image'] = image

        label = sitk.ReadImage(os.path.join(self.data_root, label_path), imageIO=r'NiftiImageIO')
        label = sitk.GetArrayFromImage(label).astype(np.float32)
        label = torch.from_numpy(label)
        data['label'] = label

        result = apply_transform(self.transforms, data)
        return {'data': result,
                'path': image_path,
                'meta_info': {'spacing': spacing,
                              'direction': direction,
                              'origin': origin}}

    def __len__(self):
        return self.dataset_size


def get_data_provider(cfg, output_folder, phase='Train', data_key=None):
    # load data json
    with open(os.path.join(cfg.DATASET.ROOT, cfg.DATASET.JSON_FILE), 'r') as f:
        dataset_json = json.load(f)

    if 'labels' in dataset_json:
        organ_map = {int(k): v for k, v in dataset_json['labels'].items()}
    else:
        organ_map = {k: str(k) for k in range(1, cfg.MODEL.GENERATOR.OUTPUT_CHANNELS)}

    # define the data augmentation transformation
    if phase.upper() == 'TRAIN':
        transform = Compose([
            AddChanneld(keys=('image', 'label')),
            Orientationd(keys=('image', 'label'), axcodes='RAS'),
            EnsureTyped(keys=('image', 'label')),
            ScaleIntensityRanged(keys=('image'),
                                 a_min=cfg.DATASET.INTENSITY_MIN, a_max=cfg.DATASET.INTENSITY_MAX,
                                 b_min=0, b_max=1,
                                 clip=cfg.DATASET.INTENSITY_CLIP),
            CropForegroundd(keys=('image', 'label'), source_key='image'),
            SpatialPadd(keys=('image', 'label'), spatial_size=cfg.DATASET.TARGET_SIZE),

            RandCropByLabelClassesd(keys=('image', 'label'),
                                    label_key='label',
                                    num_classes=cfg.DATASET.NUM_CLASSES,
                                    spatial_size=cfg.DATASET.TARGET_SIZE,
                                    num_samples=1),

            RandAffined(keys=('image', 'label'),
                        prob=cfg.DATASET.RAND_AFFINE.PROB,
                        rotate_range=cfg.DATASET.RAND_AFFINE.ROTATION_RAD_ANGLE_ZYX,
                        scale_range=cfg.DATASET.RAND_AFFINE.SCALE_RANGE_ZYX,
                        spatial_size=cfg.DATASET.TARGET_SIZE,
                        mode=['bilinear', 'nearest'],
                        padding_mode=['border', 'border'],
                        allow_missing_keys=True
                        ),

            RandGaussianSmoothd(keys=('image'),
                                prob=cfg.DATASET.RAND_GAUSSIAN_SMOOTH.PROB,
                                sigma_x=cfg.DATASET.RAND_GAUSSIAN_SMOOTH.SIGMA_X,
                                sigma_y=cfg.DATASET.RAND_GAUSSIAN_SMOOTH.SIGMA_Y,
                                sigma_z=cfg.DATASET.RAND_GAUSSIAN_SMOOTH.SIGMA_Z),

            RandScaleIntensityd(keys=('image'),
                                prob=cfg.DATASET.RAND_SCALE_INTENSITY.PROB,
                                factors=cfg.DATASET.RAND_SCALE_INTENSITY.FACTORS
                                ),

            RandShiftIntensityd(keys=('image'),
                                prob=cfg.DATASET.RAND_SHIFT_INTENSITY.PROB,
                                offsets=0.1),

            RandGaussianNoised(keys=('image'),
                               prob=cfg.DATASET.RAND_GAUSSIAN_NOISE.PROB,
                               mean=cfg.DATASET.RAND_GAUSSIAN_NOISE.MEAN,
                               std=cfg.DATASET.RAND_GAUSSIAN_NOISE.STD)
        ])

    elif phase.upper() == 'VAL':
        transform = Compose([
            AddChanneld(keys=('image', 'label')),
            Orientationd(keys=('image', 'label'), axcodes='RAS'),
            EnsureTyped(keys=('image', 'label')),
            ScaleIntensityRanged(keys=('image'),
                                 a_min=cfg.DATASET.INTENSITY_MIN, a_max=cfg.DATASET.INTENSITY_MAX,
                                 b_min=0, b_max=1,
                                 clip=cfg.DATASET.INTENSITY_CLIP),
            CropForegroundd(keys=('image', 'label'), source_key='image')
        ])
    else:
        transform = Compose([
            AddChanneld(keys=('image', 'label')),
            Orientationd(keys=('image', 'label'), axcodes='RAS'),
            EnsureTyped(keys=('image', 'label')),
            ScaleIntensityRanged(keys=('image'),
                                 a_min=cfg.DATASET.INTENSITY_MIN, a_max=cfg.DATASET.INTENSITY_MAX,
                                 b_min=0, b_max=1,
                                 clip=cfg.DATASET.INTENSITY_CLIP),
            CropForegroundd(keys=('image', 'label'), source_key='image')
        ])

    # define batch size
    batch_size = eval('cfg.{}.BATCHSIZE_PER_GPU'.format(phase.upper()))
    if torch.cuda.is_available() and phase.upper() != 'TEST':
        batch_size = batch_size * torch.cuda.device_count()

    # define the cross validation split
    if phase.upper() != 'TEST':
        data_folds = partition_dataset(dataset_json[cfg.DATASET.TRAIN_KEY],
                                       num_partitions=cfg.DATASET.CROSS_VALIDATION_FOLDERS,
                                       shuffle=True,
                                       seed=cfg.DATASET.CROSS_VALIDATION_RANDOM_SEED
                                       )
        data_list_val = select_cross_validation_folds(data_folds, folds=cfg.DATASET.CROSS_VALIDATION_CURRENT_FOLDER)
        data_list_train = [x for x in dataset_json[cfg.DATASET.TRAIN_KEY] if x not in data_list_val]

        data_list = data_list_train if phase.upper() == 'TRAIN' else data_list_val
    else:
        data_list = dataset_json[data_key]

    data_list.sort(key=lambda x: x['image'])

    data_split_output_folder = os.path.join(output_folder,
                                            f'data_split_{cfg.DATASET.CROSS_VALIDATION_CURRENT_FOLDER}_{cfg.DATASET.CROSS_VALIDATION_FOLDERS}_{cfg.DATASET.CROSS_VALIDATION_RANDOM_SEED}')
    if not os.path.exists(data_split_output_folder):
        os.makedirs(data_split_output_folder)

    # save the dataset split
    dataset_split_path = '{}.txt'.format(phase.lower()) if phase.upper() != 'TEST' else 'test_{}.txt'.format(data_key)
    with open(os.path.join(data_split_output_folder, dataset_split_path), 'w') as f:
        f.writelines([x['image'] + '\t' + x['label'] + '\n' for x in data_list])

    iteration = int(cfg.TRAIN.TOTAL_ITERATION)
    current_dataset = AMOS22Dataset(data_root=cfg.DATASET.ROOT,
                                    data_list=data_list,
                                    transforms=transform,
                                    customized_dataset_size=batch_size * iteration if phase.upper() == 'TRAIN' else 0
                                    )

    data_loader = torch.utils.data.DataLoader(current_dataset,
                                              batch_size=batch_size,
                                              shuffle=True if phase.upper() == 'TRAIN' else False,
                                              num_workers=cfg.WORKERS,
                                              pin_memory=torch.cuda.is_available())

    return data_loader, organ_map


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    data_root = r'D:\data\1_Challenge\AMOS22\AMOS22'
    data_list = 'train.txt'
    is_train = True
    is_shuffle = True

    batch_size = 2
    num_threads = 0
    is_gpu = torch.cuda.is_available()

    train_transform = Compose([
        AddChanneld(keys=('image', 'label')),
        Orientationd(keys=('image', 'label'), axcodes='RAS'),
        EnsureTyped(keys=('image', 'label')),
        ScaleIntensityRanged(keys=('image'), a_min=-676.25, a_max=200.595, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=('image', 'label'), source_key='image'),
        SpatialPadd(keys=('image', 'label'), spatial_size=(64, 160, 160)),
        # RandSpatialCropd(keys=('image', 'label'), roi_size=(64, 160, 160)),
        RandCropByLabelClassesd(keys=('image', 'label'), label_key='label', num_classes=16, spatial_size=(64, 160, 160),
                                num_samples=2),
        RandAffined(keys=['image', 'label'],
                    prob=1.0,
                    rotate_range=(0.26, 0.26, 0.26),
                    scale_range=(0.2, 0.2, 0.2),
                    spatial_size=[64, 160, 160],
                    mode=['bilinear', 'nearest'],
                    padding_mode=['border', 'border'],
                    allow_missing_keys=True
                    ),
        RandGaussianSmoothd(keys=('image'),
                            prob=0.2,
                            sigma_x=(0.5, 1.0),
                            sigma_y=(0.5, 1.0),
                            sigma_z=(0.5, 1.0)),
        RandScaleIntensityd(keys=('image'),
                            factors=0.3,
                            prob=0.5,
                            ),
        RandShiftIntensityd(keys=('image'),
                            prob=0.5,
                            offsets=0.1),
        RandGaussianNoised(keys=('image'),
                           prob=0.2,
                           mean=0.0,
                           std=0.1)
    ])

    with open(os.path.join(data_root, data_list), 'r') as f:
        data_list = [f'{os.path.sep}'.join(x.strip().split('\\')) for x in f.readlines()]

    current_dataset = AMOS22Dataset(data_root=data_root,
                                    data_list=data_list,
                                    transforms=train_transform,
                                    customized_dataset_size=0)

    train_loader = torch.utils.data.DataLoader(current_dataset,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle,
                                               num_workers=num_threads,
                                               pin_memory=is_gpu)

    for i, data in enumerate(train_loader):
        if isinstance(data['data'], list):
            current_image = [x['image'] for x in data['data']]
            current_target = [x['label'] for x in data['data']]
            current_image = torch.cat(current_image, dim=0)
            current_target = torch.cat(current_target, dim=0)
        else:
            current_image = data['data']['image']
            current_target = data['data']['label']

        current_CT = current_image[0].squeeze().detach().cpu().numpy()
        current_target = current_target[0].squeeze().detach().cpu().numpy()
        idx = np.argmax(np.sum(current_target, axis=(1, 2)))

        print("iter {}, "
              "shape: {}"
              "CT min/max: {}/{}, "
              "target min/max: {}/{}".format(i,
                                             current_CT.shape,
                                             np.min(current_CT), np.max(current_CT),
                                             np.min(current_target), np.max(current_target)))
        if True:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(current_CT[idx], cmap='gray', vmin=(-250 + 676) / 876, vmax=(400 + 676) / 876)
            plt.subplot(1, 2, 2)
            plt.imshow(current_target[idx], cmap='gray')
            plt.show()

    print('Congrats! May the force be with you ...')
