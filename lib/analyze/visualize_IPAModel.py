#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 9/27/2019 12:00 PM
# FILE: visualize.py

# sys
import os
import numpy as np
from skimage.measure import find_contours
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# torch
import torch

# project
from lib.analyze.base_visualize import VisualizationAnalysis


def visualize(model,
              iteration,
              output_dir,
              display_frequency,
              window_size=256):
    # update website
    visualizer = VisualizationAnalysis(model, iteration, output_dir)
    webpage = visualizer.webpage

    before_min = 0
    before_max = 2000
    after_min = 0
    after_max = 1
    min_v, max_v = 1000 - 160, 1000 + 240
    threshold = 0

    scale = (before_max - before_min) / (after_max - after_min)
    shift = before_min - after_min * scale

    input = visualizer.tensor2image(model.input, scale, shift)
    target = model.target.squeeze().detach().cpu().numpy()
    target = {k: [target == k, v] for k, v in zip([6, 7], ['IPA_L', 'IPA_R'])}

    output_activation = model.output[0].squeeze().detach().cpu().numpy()
    output = output_activation > threshold

    for current_organ_index, current_target_mask in target.items():
        for current_view, current_reduction_axis in zip(['axial', 'coronal', 'sagittal'], [(-1, -1), (0, -1), (0, 0)]):
            # define the output path
            img_path = 'iter_{:>08}_{}_{}'.format(iteration, current_target_mask[1], current_view)
            output_path = os.path.join(output_dir, 'web', 'images', img_path)

            # get the slice index with maximum square on the target
            current_max_index = np.argmax((np.sum(np.sum(current_target_mask[0],
                                                         axis=current_reduction_axis[0]),
                                                  axis=current_reduction_axis[1])))

            # get the input/output/target for the specific slice/view
            if current_view == 'sagittal':
                current_input_slice = input[:, :, current_max_index]
                current_output_slice = output[current_organ_index][:, :, current_max_index]
                current_target_slice =current_target_mask[0][:, :, current_max_index]
            elif current_view == 'axial':
                current_input_slice = input[current_max_index, :, :]
                current_output_slice = output[current_organ_index][current_max_index, :, :]
                current_target_slice = current_target_mask[0][current_max_index, :, :]
            elif current_view == 'coronal':
                current_input_slice = input[:, current_max_index, :]
                current_output_slice = output[current_organ_index][:, current_max_index, :]
                current_target_slice = current_target_mask[0][:, current_max_index, :]
            else:
                raise ValueError("Unexpected view: {}".format(current_view))

            # calculate the contour
            output_contour = find_contours(current_output_slice, 0.5)
            target_contour = find_contours(current_target_slice, 0.5)

            # display and save the contour
            plt.figure()
            plt.imshow(current_input_slice, cmap='gray', vmin=min_v, vmax=max_v)
            for current_contours, current_color in zip([output_contour], ['y']):
                for current_contour in current_contours:
                    plt.plot(current_contour[:, 1], current_contour[:, 0], linestyle='-', linewidth=1,
                             color=current_color)

            for current_contours, current_color in zip([target_contour], ['g']):
                for current_contour in current_contours:
                    plt.plot(current_contour[:, 1], current_contour[:, 0], linestyle='-', linewidth=1,
                             color=current_color)

            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

    all_saved_images_list = os.listdir(os.path.join(output_dir, 'web', 'images'))
    all_saved_images_dict = {}
    for current_image in all_saved_images_list:
        current_iter = int(current_image[5:13])
        if current_iter not in all_saved_images_dict:
            all_saved_images_dict[current_iter] = [current_image]
        else:
            all_saved_images_dict[current_iter].append(current_image)

    for n in range(iteration, -1, -display_frequency):
    #for n in range(iteration, iteration - 2, -display_frequency):
        webpage.add_header('iteration_{:>08}'.format(n))
        ims, txts, links = [], [], []
        if n not in all_saved_images_dict:
            continue
        for current_image_path in all_saved_images_dict[n]:
            #img_path = 'iter_{:>08}_{}{}'.format(n, current_image_name, '.png')
            img_path = current_image_path
            current_image_name = '_'.join(os.path.basename(current_image_path).split('_')[2:])[:-4]
            ims.append(img_path)
            txts.append(current_image_name)
            links.append(img_path)
        webpage.add_images(ims, txts, links, window_size)
    webpage.save()
