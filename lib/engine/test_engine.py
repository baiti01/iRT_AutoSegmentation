#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 8/12/2019 3:43 PM
# FILE: train_engine.py

# sys
import logging
import os
import pickle as pkl
import SimpleITK as sitk
import numpy as np

# torch
import torch

# monai
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import KeepLargestConnectedComponent

# project
from lib.utils.analyze import dataset_performance_analysis, result_visualization_and_excel_generation, \
    organ_result_generation


def save(data, meta_info, output_folder, organ_name):
    data = (data.squeeze().detach().cpu().numpy() > 0).astype(np.int16)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(meta_info['spacing'])
    image.SetDirection(meta_info['direction'])
    image.SetOrigin(meta_info['origin'])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sitk.WriteImage(image, os.path.join(output_folder, f'{organ_name}.nii.gz'))


def do_test(test_loader,
            model,
            cfg,
            visualize,
            final_output_dir):
    model.eval()

    all_result = {}
    for i, current_data in enumerate(test_loader):
        patient_result = {}

        current_patient_path = current_data['path'][0]

        # generate the input CT image
        current_image = current_data['data']['image']
        current_image = current_image.to('cuda' if torch.cuda.is_available() else 'cpu')

        # generate the auto-segmentation result
        with torch.no_grad():
            current_auto_segmentation_result = sliding_window_inference(inputs=current_image,
                                                                        roi_size=cfg.DATASET.TARGET_SIZE,
                                                                        sw_batch_size=1,
                                                                        predictor=model.generator,
                                                                        mode='gaussian',
                                                                        overlap=cfg.VAL.OVERLAP_RATIO,
                                                                        sw_device='cuda' if torch.cuda.is_available() else 'cpu',
                                                                        device='cpu')
            current_auto_segmentation_result = current_auto_segmentation_result.detach()

        # generate the GT labels
        if cfg.DATASET.NAME == 'HN_end2end':
            GT_labels = {k: v for k, v in current_data['data'].items() if
                         k in ['MASK_P1', 'MASK_P2', 'MASK_P3', 'MASK_P4']}
        else:
            GT_labels = {k: v for k, v in current_data['data'].items() if k in ['label']}

        current_data = decollate_batch(current_data)[0]
        for k, current_GT_label in GT_labels.items():
            current_unique_ids = torch.unique(current_GT_label)[1:]

            for current_organ_id in current_unique_ids:
                current_organ_id = current_organ_id.long().item()
                current_organ_name = model.organ_map[current_organ_id]
                model.logger.info(current_patient_path + '\t' + current_organ_name)
                current_organ_label = (current_GT_label == current_organ_id).float()

                current_auto_segmentation_mask = current_auto_segmentation_result[:, current_organ_id].unsqueeze(1)

                # calculate the metric
                patient_result[current_organ_name] = organ_result_generation(
                    reference_mask=current_organ_label.squeeze().detach().cpu().numpy(),
                    predict_mask=current_auto_segmentation_mask[0].squeeze().detach().cpu().numpy() > 0,
                    spacing=current_data['meta_info']['spacing'])

                # visualization
                if cfg.TEST.IS_VISUALIZATION:
                    dataset = cfg.DATASET.NAME
                    patient_ID = os.path.basename(current_patient_path)
                    model.visualization_info = {'dataset': dataset,
                                                'patient_ID': patient_ID,
                                                'organ_name': current_organ_name,
                                                'input': current_image,
                                                'label': current_organ_label,
                                                'prediction': current_auto_segmentation_mask}
                    visualize(model, i, os.path.join(final_output_dir, "test"), 1)

                # save the prediction results
                if cfg.TEST.SAVE_PREDICTION:
                    save(current_auto_segmentation_mask,
                         current_data['meta_info'],
                         output_folder=os.path.join(final_output_dir, 'test', 'predictions', 'auto',
                                                    current_patient_path),
                         organ_name=current_organ_name)

        all_result[current_patient_path] = patient_result

    model.logger.info('Saving the result ...')
    if not os.path.exists(os.path.join(final_output_dir, 'test', 'performance')):
        os.makedirs(os.path.join(final_output_dir, 'test', 'performance'))
    with open(os.path.join(final_output_dir, 'test', 'performance', 'metric_result.pkl'), 'wb') as f:
        pkl.dump(all_result, f)

    model.logger.info('Analyzing the result ...')
    dataset_performance_result = dataset_performance_analysis(result=all_result)
    result_visualization_and_excel_generation(result=dataset_performance_result,
                                              output_folder=os.path.join(final_output_dir, 'test',
                                                                         'performance'),
                                              prefix='auto')
