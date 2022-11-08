#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 9/27/2019 12:00 PM
# FILE: visualize.py

# sys
import os
import matplotlib
import numpy as np
from skimage.measure import find_contours
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project
from lib.analyze.base_visualize import VisualizationAnalysis


def Calculate_DSC(prediction, groundtruth):
    DSC = 2 * np.sum(prediction * groundtruth) / (np.sum(prediction) + np.sum(groundtruth) + 1e-5) * 100
    return DSC


def plot_contours(image, target, prediction_contour, min_v, max_v, output_path):
    current_prediction_DSC = Calculate_DSC(prediction=target, groundtruth=prediction_contour)

    # calculate the contour
    prediction_contour = find_contours(prediction_contour, 0.5)
    target_contour = find_contours(target, 0.5)

    # display and save the contour
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=min_v, vmax=max_v)
    for current_contours, current_color in zip([prediction_contour, target_contour], ['r', 'g']):
        for current_contour in current_contours:
            plt.plot(current_contour[:, 1], current_contour[:, 0], linestyle='-', linewidth=1,
                     color=current_color)
    plt.axis('off')
    plt.savefig('{}_{:.1f}.png'.format(output_path, current_prediction_DSC), bbox_inches='tight')
    plt.close()


def visualize(model,
              iteration,
              output_dir,
              display_frequency,
              table_columns=None,
              window_size=256):
    # update website
    visualizer = VisualizationAnalysis(model, iteration, output_dir, table_columns=table_columns)
    webpage = visualizer.webpage

    before_min = 0
    before_max = 2000
    after_min = 0
    after_max = 1
    min_v, max_v = 1000 - 160, 1000 + 240
    threshold = 0

    scale = (before_max - before_min) / (after_max - after_min)
    shift = before_min - after_min * scale

    # setup the input
    input = visualizer.tensor2image(model.visualization_info['input'], scale, shift)

    organ_name = model.visualization_info['organ_name']
    # we assume the target has size of B*C*D*H*W,
    # where C denotes the number of classes, including the background (id = 0)
    b, c, _, _, _ = model.visualization_info['label'].shape

    target = model.visualization_info['label'].squeeze().detach().cpu().numpy()

    prediction_output_activation = model.visualization_info['prediction'].squeeze().detach().cpu().numpy()
    prediction_output = prediction_output_activation > threshold

    output_folder = os.path.join(output_dir, 'web', 'images', organ_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for current_view, current_reduction_axis in zip(['axial', 'coronal', 'sagittal'], [(1, 2), (0, 2), (0, 1)]):
        # get the slice index with maximum square on the target or output if the target is empty
        current_max_index = np.argmax(np.sum(target, axis=current_reduction_axis))

        # get the input/output/target for the specific slice/view
        if current_view == 'axial':
            # for axial, display 5 images, top/bottom 2 images and max size image
            # max size
            try:
                current_non_zero_ids = np.where(np.sum(target, axis=current_reduction_axis))[0]
                current_slice_ids = {'Inf1': current_non_zero_ids[0],
                                     'Middle': current_non_zero_ids[len(current_non_zero_ids) // 2],
                                     'Sup1': current_non_zero_ids[-1],
                                     }

                if len(current_non_zero_ids) >= 3:
                    current_slice_ids['Inf2'] = current_non_zero_ids[1]
                    current_slice_ids['Sup2'] = current_non_zero_ids[-2]
            except:
                current_slice_ids = {'max': current_max_index}

            for current_location, current_slice_idx in current_slice_ids.items():
                current_input_slice = input[current_slice_idx, :, :]
                current_prediction_output_slice = prediction_output[current_slice_idx, :, :]
                current_target_slice = target[current_slice_idx, :, :]

                image_path = 'iter_{:>08}_{}_{}_{}'.format(iteration,
                                                           model.visualization_info['patient_ID'],
                                                           current_view,
                                                           current_location)
                plot_contours(image=current_input_slice,
                              target=current_target_slice,
                              prediction_contour=current_prediction_output_slice,
                              min_v=min_v,
                              max_v=max_v,
                              output_path=os.path.join(output_folder, image_path))

        elif current_view == 'coronal':
            current_input_slice = input[:, current_max_index, :][::-1]
            current_prediction_output_slice = prediction_output[:, current_max_index, :][::-1]
            current_target_slice = target[:, current_max_index, :][::-1]

            image_path = 'iter_{:>08}_{}_{}'.format(iteration,
                                                    model.visualization_info['patient_ID'],
                                                    current_view)
            plot_contours(image=current_input_slice,
                          target=current_target_slice,
                          prediction_contour=current_prediction_output_slice,
                          min_v=min_v,
                          max_v=max_v,
                          output_path=os.path.join(output_folder, image_path))

        elif current_view == 'sagittal':
            current_input_slice = input[:, :, current_max_index][::-1]
            current_prediction_output_slice = prediction_output[:, :, current_max_index][::-1]
            current_target_slice = target[:, :, current_max_index][::-1]

            image_path = 'iter_{:>08}_{}_{}'.format(iteration,
                                                    model.visualization_info['patient_ID'],
                                                    current_view)
            plot_contours(image=current_input_slice,
                          target=current_target_slice,
                          prediction_contour=current_prediction_output_slice,
                          min_v=min_v,
                          max_v=max_v,
                          output_path=os.path.join(output_folder, image_path))
        else:
            raise ValueError("Unexpected view: {}".format(current_view))

    all_saved_images_list = os.listdir(output_folder)
    all_saved_images_dict = {}
    for current_image in all_saved_images_list:
        current_iter = int(current_image[5:13])
        if current_iter not in all_saved_images_dict:
            all_saved_images_dict[current_iter] = [os.path.join(organ_name, current_image)]
        else:
            all_saved_images_dict[current_iter].append(os.path.join(organ_name, current_image))

    for n in range(iteration, -1, -display_frequency):
        # for n in range(iteration, iteration - 2, -display_frequency):
        webpage.add_header('iteration_{:>08}'.format(n))
        ims, txts, links = [], [], []
        if n not in all_saved_images_dict:
            continue
        for current_image_path in all_saved_images_dict[n]:
            # img_path = 'iter_{:>08}_{}{}'.format(n, current_image_name, '.png')
            img_path = current_image_path
            current_image_name = '_'.join(os.path.basename(current_image_path).split('_')[2:])[:-4]
            ims.append(img_path)
            txts.append(current_image_name)
            links.append(img_path)
        webpage.add_images(ims, txts, links, window_size)
    webpage.save(prefix=organ_name)
