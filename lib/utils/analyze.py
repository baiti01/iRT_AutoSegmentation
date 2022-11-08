#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 3/2/2022

import os
import pickle as pkl

import SimpleITK as sitk
import matplotlib
import numpy as np
import pandas as pd
import surface_distance.metrics as surface_metrics

matplotlib.use('Agg')
import matplotlib.pyplot as plt

font = {'family': 'normal',
        'weight': 'bold',
        'size': 12}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **{'labelweight': 'bold'})

import seaborn as sns


def metrics_calculation(gt_mask, pred_mask, tolerance=(1.5, 2.5, 3.5), spacing_mm=(1.17, 1.17)):
    gt_mask_bool = gt_mask.astype(bool)
    pred_mask_bool = pred_mask.astype(bool)

    # DSC
    DSC = surface_metrics.compute_dice_coefficient(mask_gt=gt_mask_bool, mask_pred=pred_mask_bool)

    surface_distance = surface_metrics.compute_surface_distances(mask_gt=gt_mask_bool,
                                                                 mask_pred=pred_mask_bool,
                                                                 spacing_mm=spacing_mm)

    # surface distance with different tolerances
    surface_distance_dict = {}
    for current_tolerance in tolerance:
        surface_distance_dict[current_tolerance] = surface_metrics.compute_surface_dice_at_tolerance(
            surface_distance,
            tolerance_mm=current_tolerance)

    # HD95
    HD95 = surface_metrics.compute_robust_hausdorff(surface_distance, 95)

    # TODO: 1) add path length; 2) organ specific tolerance

    #  add path length

    return {'HD95': HD95, 'DSC': DSC, 'surface_distance': surface_distance_dict}


def organ_result_generation(reference_mask, predict_mask, spacing):
    resolution_x, resolution_y, thickness = spacing
    assert predict_mask.shape == reference_mask.shape

    # compare between reference mask and algorithm AI mask
    metric_result = {'common_slices': {},
                     'missing_slices': {},
                     '3D_DSC': -1,
                     'over_segmented_slices': {'continued_slices': [],
                                               'all_slices': []},
                     '3D_DSC_Vanilla': -1
                     }
    # part 1: common slices between the AI and physician contours
    # part 1, step 1: find the common index
    common_slice_idx = np.nonzero(np.sum(predict_mask, axis=(1, 2)) * np.sum(reference_mask, axis=(1, 2)))[0]

    # part 1, step 2: calculate the metrics for each 2D slice
    for current_slice_idx in common_slice_idx:
        current_metric = metrics_calculation(gt_mask=reference_mask[current_slice_idx],
                                             pred_mask=predict_mask[current_slice_idx],
                                             spacing_mm=(resolution_x, resolution_y))
        metric_result['common_slices'][current_slice_idx] = current_metric

    # part 2: missing slices (physician has the contour but the AI doesn't)
    # TODO: calculate the add path length for the missing slices
    reference_non_empty_slice_idx = np.nonzero(np.sum(reference_mask, axis=(1, 2)))[0]
    missing_slice_idx = list(set(reference_non_empty_slice_idx).difference(set(common_slice_idx)))
    missing_slice_idx.sort()
    for current_slice_idx in missing_slice_idx:
        metric_result['missing_slices'][current_slice_idx] = {'APL': -1}

    # part 3: over segmented slices defined as those slices: has foreground in the prediction while no foreground in the reference
    # case 1: only consider those slices connected with the common slices since the other potential slices are likely due to wrong segmentation
    pred_non_empty_slice_idx = np.nonzero(np.sum(predict_mask, axis=(1, 2)))[0]
    if len(common_slice_idx) > 0:
        for direction, current_start_stop_slice_idx in zip([-1, 1], [common_slice_idx[0], common_slice_idx[-1]]):
            is_continue = True
            while is_continue:
                current_start_stop_slice_idx = current_start_stop_slice_idx + direction
                if current_start_stop_slice_idx in pred_non_empty_slice_idx:
                    metric_result['over_segmented_slices']['continued_slices'].append(current_start_stop_slice_idx)
                else:
                    is_continue = False
        metric_result['over_segmented_slices']['continued_slices'].sort()

    # case 2: consider all the over-segmented slices
    over_segmented_slice_idx = list(set(pred_non_empty_slice_idx).difference(set(common_slice_idx)))
    over_segmented_slice_idx.sort()
    metric_result['over_segmented_slices']['all_slices'] = over_segmented_slice_idx

    # part 4: 3D DSC based on the non-empty target slices
    metric_result['3D_DSC'] = surface_metrics.compute_dice_coefficient(
        mask_gt=reference_mask[reference_non_empty_slice_idx].astype(bool),
        mask_pred=predict_mask[reference_non_empty_slice_idx].astype(bool))

    # part 5: vanilla 3D DSC based on the non-empty target slices
    metric_result['3D_DSC_Vanilla'] = surface_metrics.compute_dice_coefficient(
        mask_gt=reference_mask.astype(bool),
        mask_pred=predict_mask.astype(bool))

    return metric_result


def patient_result_generation(data_path, patient_MRN):
    all_files = os.listdir(os.path.join(data_path, patient_MRN))
    clinical_AI_organ_names = [x.split('clinical_AI_')[-1][:-7] for x in all_files if x.startswith('clinical_AI')]

    patient_result = {}
    for current_organ_name in clinical_AI_organ_names:
        if 'Body' in current_organ_name:
            continue

        # read the clinical AI result
        current_clinical_AI_name = f'clinical_AI_{current_organ_name}.nii.gz'
        current_clinical_AI_path = os.path.join(data_path, patient_MRN, current_clinical_AI_name)
        current_clinical_AI_organ = sitk.ReadImage(current_clinical_AI_path)
        current_clinical_AI_organ = sitk.GetArrayFromImage(current_clinical_AI_organ).astype(np.float32)

        # exclusion criterion 1: empty clinica AI contour
        if np.sum(current_clinical_AI_organ) == 0:
            logging.warning('Empty AI organ: {}'.format(current_clinical_AI_path))
            continue

        # read algorithm AI result
        current_algorithm_AI_name = f'algorithm_AI_{current_organ_name}.nii.gz'
        current_algorithm_AI_path = os.path.join(data_path, patient_MRN, current_algorithm_AI_name)

        # exclusion criterion 2: No algorithm AI contour
        if not os.path.exists(current_algorithm_AI_path):
            logging.warning('Cannot find the associated algorithm AI contour: {}'.format(current_algorithm_AI_path))
            continue

        current_algorithm_AI_organ = sitk.ReadImage(current_algorithm_AI_path)
        current_algorithm_AI_organ = sitk.GetArrayFromImage(current_algorithm_AI_organ).astype(np.float32)

        # read the physician contour
        current_physician_name = f'physician_{current_organ_name}.nii.gz'
        current_physician_path = os.path.join(data_path, patient_MRN, current_physician_name)

        # exclusion criterion 3: No physician contour
        if not os.path.exists(current_physician_path):
            logging.warning('Cannot find the associated physician contour: {}'.format(current_clinical_AI_path))
            continue

        current_physician_organ = sitk.ReadImage(current_physician_path)
        spacing = current_physician_organ.GetSpacing()
        current_physician_organ = sitk.GetArrayFromImage(current_physician_organ).astype(np.float32)

        # exclusion criterion 4: the physician didn't use this AI contour
        if np.sum(current_clinical_AI_organ - current_physician_organ) == 0:
            logging.warning('The physician did not use the AI contour: {}'.format(current_physician_path))
            continue

        # calculate the metric and write into the final dictionary
        patient_result.setdefault(current_organ_name, {})
        current_organ_result = organ_result_generation(reference_mask=current_physician_organ,
                                                       predict_mask=current_algorithm_AI_organ,
                                                       spacing=spacing)
        patient_result[current_organ_name] = current_organ_result
    return patient_MRN, patient_result


def dataset_result_generation(data_path, output_path):
    # folder structure:
    # each patient has a folder named as MRN
    # inside each folder, you have AI_organ_name.nii.gz and physician_organ_name.nii.gz

    all_results = {}

    def save_result(result):
        all_results[result[0]] = result[1]

    # pool = mp.Pool(8)
    for current_patient_MRN in os.listdir(data_path):
        # pool.apply_async(patient_result_generation, args=(data_path, current_patient_MRN), callback=save_result)
        current_patient_result = patient_result_generation(data_path, current_patient_MRN)
        save_result(current_patient_result)

    # pool.close()
    # pool.join()

    with open(output_path, 'wb') as f:
        pkl.dump(all_results, f)

    return all_results


def common_slices_analysis(result):
    all_result = {'Inf/Sup': {}, 'Middle': {}}

    # define the start/stop slices of the middle part based on the algorithm AI
    slice_idxs = list(result.keys())
    if len(slice_idxs) < 4:
        top_idx = 1
        bottom_idx = len(slice_idxs) - 2
    else:
        top_idx = min(len(slice_idxs) // 4, 3)
        bottom_idx = len(slice_idxs) - 1 - top_idx

    # record the metrics
    for i, current_slice_idx in enumerate(slice_idxs):
        if (i < top_idx) or (i > bottom_idx):
            location = 'Inf/Sup'
        else:
            location = 'Middle'

        # increase the slice number
        all_result[location].setdefault('slice_number', 0)
        all_result[location]['slice_number'] += 1

        current_slice_metric = result[current_slice_idx]

        # increase the un-revised slice number
        all_result[location].setdefault('unrevised_slice_number', 0)
        if current_slice_metric['HD95'] == 0:
            all_result[location]['unrevised_slice_number'] += 1

        # record all the metrics
        for current_metric_name, current_metric_value in current_slice_metric.items():
            if 'surface_distance' == current_metric_name:
                for surface_distance_target, surface_distance_value in current_metric_value.items():
                    all_result[location].setdefault(f'{current_metric_name}_{surface_distance_target}', [])
                    all_result[location][f'{current_metric_name}_{surface_distance_target}'].append(
                        surface_distance_value)
            else:
                all_result[location].setdefault(current_metric_name, [])
                all_result[location][current_metric_name].append(current_metric_value)

    for current_location_name, current_location_value in all_result.items():
        if len(current_location_value) == 0:
            all_result[current_location_name] = {'slice_number': 0, 'unrevised_slice_number': 0}

    return all_result


def dataset_performance_analysis(result):
    if not isinstance(result, dict):
        try:
            with open(result, 'rb') as f:
                result = pkl.load(f)
        except:
            raise ValueError('Check the input! It should be either a python dictionary object or a path!')

    new_result = {}
    for current_patient_name, current_patient_value in result.items():
        for current_organ_name, current_organ_value in current_patient_value.items():
            if current_organ_name not in new_result:
                new_result[current_organ_name] = {}
            new_result[current_organ_name][current_patient_name] = current_organ_value
    result = new_result

    all_result = {}
    for organ_name, organ_value in result.items():
        all_result.setdefault(organ_name, {'common_slices': {},
                                           'missing_slices': {},
                                           '3D_DSC': {},
                                           '3D_DSC_Vanilla': {},
                                           'over_segmented_slices': {}})
        for patient_MRN, patient_value in organ_value.items():
            # step 1: analyze the missing slices
            missing_slices_result = patient_value['missing_slices']
            missing_slices_APL_total = [v['APL'] for _, v in missing_slices_result.items()]
            missing_slices_APL_total = sum(missing_slices_APL_total)
            all_result[organ_name]['missing_slices'][patient_MRN] = {'missing_slice_number': len(missing_slices_result),
                                                                     'APL_total': missing_slices_APL_total}

            # step 2: analyze the over-segmented slices
            over_segmented_slices_result = patient_value['over_segmented_slices']
            all_result[organ_name]['over_segmented_slices'][patient_MRN] = {
                'continued_slices': len(over_segmented_slices_result['continued_slices']),
                'all_slices': len(over_segmented_slices_result['all_slices'])}

            # step 2: analyze the overall performance in 3D
            all_result[organ_name]['3D_DSC'][patient_MRN] = patient_value['3D_DSC']

            # step 3: analyze the overall performance in vanilla 3D
            all_result[organ_name]['3D_DSC_Vanilla'][patient_MRN] = patient_value['3D_DSC_Vanilla']

            # step 4: analyze the slice-wise performance
            all_result[organ_name]['common_slices'][patient_MRN] = common_slices_analysis(
                patient_value['common_slices'])

    return all_result


def set_figure_style(fig, ax, ylabel, output_path, legend=None, legend_location=4):
    ax.yaxis.grid(True, linestyle='--')
    ax.set(xlabel=None)
    ax.set(ylabel=ylabel)

    if legend is not None:
        ax.legend(loc=legend_location)

    plt.xticks(rotation=90)
    plt.savefig(output_path, bbox_inches='tight', transparent=False)
    plt.close(fig)


def get_data_frame(result, metric):
    inf_sup_result = {k: {k1: v1['Inf/Sup'][metric] for k1, v1 in v['common_slices'].items() if metric in v1['Inf/Sup']}
                      for k, v in result.items()}
    middle_result = {k: {k1: v1['Middle'][metric] for k1, v1 in v['common_slices'].items() if metric in v1['Middle']}
                     for k, v in result.items()}

    for current_organ, current_organ_value in inf_sup_result.items():
        current_organ_value = [v1 for k, v1 in current_organ_value.items()]
        current_organ_value = sum(current_organ_value, [])
        inf_sup_result[current_organ] = {metric: current_organ_value}

    for current_organ, current_organ_value in middle_result.items():
        current_organ_value = [v1 for k, v1 in current_organ_value.items()]
        current_organ_value = sum(current_organ_value, [])
        middle_result[current_organ] = {metric: current_organ_value}

    inf_sup_result = pd.DataFrame.from_dict(inf_sup_result).T
    inf_sup_result['location'] = 'Inf/Sup'
    inf_sup_result = inf_sup_result.explode(metric)

    middle_result = pd.DataFrame.from_dict(middle_result).T
    middle_result['location'] = 'Middle'
    middle_result = middle_result.explode(metric)

    all_result_df = pd.concat([inf_sup_result, middle_result]).reset_index()

    return all_result_df


def result_visualization_and_excel_generation(result, output_folder, prefix):
    image_output_folder = os.path.join(output_folder, prefix, 'images')
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    excel_data_frame = {}

    # TODO: visualize the APL

    # visualize the 3D DSC
    DSC_3D_result = {k: v['3D_DSC'] for k, v in result.items()}
    DSC_3D_result = pd.DataFrame.from_dict(DSC_3D_result)
    fig, ax = plt.subplots()
    DSC_3D_result.boxplot()
    set_figure_style(fig, ax, ylabel='3D DSC', output_path=os.path.join(image_output_folder, 'DSC_3D.png'))

    DSC_3D_mean = DSC_3D_result.mean().rename('mean').to_frame()
    DSC_3D_std = DSC_3D_result.std().rename('std').to_frame()
    excel_data_frame['DSC_3D'] = pd.concat([DSC_3D_mean, DSC_3D_std], axis=1)

    # visualize the vanilla 3D DSC
    DSC_vanilla_3D_result = {k: v['3D_DSC_Vanilla'] for k, v in result.items()}
    DSC_vanilla_3D_result = pd.DataFrame.from_dict(DSC_vanilla_3D_result)
    fig, ax = plt.subplots()
    DSC_vanilla_3D_result.boxplot()
    set_figure_style(fig, ax, ylabel='Vanilla 3D DSC',
                     output_path=os.path.join(image_output_folder, 'vanilla_DSC_3D.png'))

    DSC_vanilla_3D_mean = DSC_vanilla_3D_result.mean().rename('mean').to_frame()
    DSC_vanilla_3D_std = DSC_vanilla_3D_result.std().rename('std').to_frame()
    excel_data_frame['3D_DSC_Vanilla'] = pd.concat([DSC_vanilla_3D_mean, DSC_vanilla_3D_std], axis=1)

    # visualize the percentage of missing slices
    missing_slices = {k: {k1: v1['missing_slice_number'] for k1, v1 in v['missing_slices'].items()} for k, v in
                      result.items()}
    missing_slices = pd.DataFrame.from_dict(missing_slices)

    inf_sup_slices = {k: {k1: v1['Inf/Sup']['slice_number'] for k1, v1 in v['common_slices'].items()} for k, v in
                      result.items()}
    inf_sup_slices = pd.DataFrame.from_dict(inf_sup_slices)

    middle_slices = {k: {k1: v1['Middle']['slice_number'] for k1, v1 in v['common_slices'].items()} for k, v in
                     result.items()}
    middle_slices = pd.DataFrame.from_dict(middle_slices)

    total_slices = missing_slices + inf_sup_slices + middle_slices

    missing_slices_percentage = missing_slices / total_slices
    fig, ax = plt.subplots()
    missing_slices_percentage.boxplot()
    set_figure_style(fig, ax, ylabel='Missing Slices Percentage',
                     output_path=os.path.join(image_output_folder, 'missing_slices_percentage.png'))

    missing_slices_percentage_mean = missing_slices_percentage.mean().rename('mean').to_frame()
    missing_slices_percentage_std = missing_slices_percentage.std().rename('std').to_frame()
    excel_data_frame['missing_slices_percentage'] = pd.concat(
        [missing_slices_percentage_mean, missing_slices_percentage_std], axis=1)

    # visualize the percentage of the over-segmented slices
    continuous_over_segmented_slices = {k: {k1: v1['continued_slices'] for k1, v1 in v['over_segmented_slices'].items()}
                                        for k, v in
                                        result.items()}
    continuous_over_segmented_slices = pd.DataFrame.from_dict(continuous_over_segmented_slices)
    continuous_over_segmented_slices_percentage = continuous_over_segmented_slices / (
                continuous_over_segmented_slices + inf_sup_slices + middle_slices)
    fig, ax = plt.subplots()
    continuous_over_segmented_slices_percentage.boxplot()
    set_figure_style(fig, ax, ylabel='Continuous Over-Seg Slices Percentage',
                     output_path=os.path.join(image_output_folder, 'continuous_over_segmented_slices_percentage.png'))

    continuous_over_segmented_slices_mean = continuous_over_segmented_slices_percentage.mean().rename('mean').to_frame()
    continuous_over_segmented_slices_std = continuous_over_segmented_slices_percentage.std().rename('std').to_frame()
    excel_data_frame['cont_over_seg_percentage'] = pd.concat(
        [continuous_over_segmented_slices_mean, continuous_over_segmented_slices_std], axis=1)

    all_over_segmented_slices = {k: {k1: v1['all_slices'] for k1, v1 in v['over_segmented_slices'].items()} for k, v in
                                 result.items()}
    all_over_segmented_slices = pd.DataFrame.from_dict(all_over_segmented_slices)
    all_over_segmented_slices_percentage = all_over_segmented_slices / (
                all_over_segmented_slices + inf_sup_slices + middle_slices)
    fig, ax = plt.subplots()
    all_over_segmented_slices_percentage.boxplot()
    set_figure_style(fig, ax, ylabel='All Over-Seg Slices Percentage',
                     output_path=os.path.join(image_output_folder, 'all_over_segmented_slices_percentage.png'))

    all_over_segmented_slices_mean = all_over_segmented_slices_percentage.mean().rename('mean').to_frame()
    all_over_segmented_slices_std = all_over_segmented_slices_percentage.std().rename('std').to_frame()
    excel_data_frame['all_over_seg_percentage'] = pd.concat(
        [all_over_segmented_slices_mean, all_over_segmented_slices_std], axis=1)

    # add the un-revised ratio
    un_revised_inf_sup_slices = {
        k: {k1: v1['Inf/Sup']['unrevised_slice_number'] for k1, v1 in v['common_slices'].items()} for k, v in
        result.items()}
    un_revised_inf_sup_slices = pd.DataFrame.from_dict(un_revised_inf_sup_slices)
    un_revised_inf_sup_slices_ratio = un_revised_inf_sup_slices / inf_sup_slices
    un_revised_inf_sup_slices_ratio_mean = un_revised_inf_sup_slices_ratio.mean().rename('mean_Inf/Sup').to_frame()
    un_revised_inf_sup_slices_ratio_std = un_revised_inf_sup_slices_ratio.std().rename('std_Inf/Sup').to_frame()

    un_revised_middle_slices = {k: {k1: v1['Middle']['unrevised_slice_number'] for k1, v1 in v['common_slices'].items()}
                                for k, v in result.items()}
    un_revised_middle_slices = pd.DataFrame.from_dict(un_revised_middle_slices)
    un_revised_middle_slices_ratio = un_revised_middle_slices / middle_slices
    un_revised_middle_slices_ratio_mean = un_revised_middle_slices_ratio.mean().rename('mean_Middle').to_frame()
    un_revised_middle_slices_ratio_std = un_revised_middle_slices_ratio.std().rename('std_Middle').to_frame()

    un_revised_total_ratio = (un_revised_inf_sup_slices + un_revised_middle_slices) / total_slices
    un_revised_total_ratio_mean = un_revised_total_ratio.mean().rename('mean_All').to_frame()
    un_revised_total_ratio_std = un_revised_total_ratio.std().rename('std_All').to_frame()

    excel_data_frame['unrevised_ratio'] = pd.concat([un_revised_total_ratio_mean,
                                                     un_revised_middle_slices_ratio_mean,
                                                     un_revised_inf_sup_slices_ratio_mean,
                                                     un_revised_total_ratio_std,
                                                     un_revised_middle_slices_ratio_std,
                                                     un_revised_inf_sup_slices_ratio_std], axis=1)

    # visualize the common slices
    # get the metrics
    metrics = result[list(result.keys())[0]]['common_slices']
    metrics = metrics[list(metrics.keys())[0]]['Middle']
    metrics = [k for k, v in metrics.items() if isinstance(v, list)]

    for current_metric in metrics:
        current_metric_data_frame = get_data_frame(result, current_metric)

        legend_location = 4
        if 'surface' in current_metric:
            target = current_metric.split('_')[-1]
            ylabel = f'Surface DSC ({target}mm)'
            legend_location = 4
        elif current_metric == 'DSC':
            ylabel = 'Slice DSC'
            legend_location = 4
        else:
            ylabel = 'HD95 (mm)'
            legend_location = 1

        # all data
        fig, ax = plt.subplots()
        sns.boxplot(ax=ax, x='index', y=current_metric,
                    data=current_metric_data_frame,
                    showfliers=False)
        set_figure_style(fig, ax, ylabel=ylabel,
                         output_path=os.path.join(image_output_folder, f'{current_metric}_all.png'),
                         legend_location=legend_location)

        # split data
        fig, ax = plt.subplots()
        sns.boxplot(ax=ax, x='index', y=current_metric,
                    data=current_metric_data_frame,
                    hue='location',
                    showfliers=False)
        set_figure_style(fig, ax, ylabel=ylabel,
                         output_path=os.path.join(image_output_folder, f'{current_metric}_location.png'),
                         legend_location=legend_location)

        # excel frame
        current_metric_data_frame_mean = current_metric_data_frame.groupby(['index', 'location'])[
            current_metric].mean().swaplevel()
        current_metric_data_frame_std = current_metric_data_frame.groupby(['index', 'location'])[
            current_metric].std().swaplevel()
        current_metric_data_frame_mean_all = current_metric_data_frame.groupby(['index'])[current_metric].mean()
        current_metric_data_frame_std_all = current_metric_data_frame.groupby(['index'])[current_metric].std()
        current_metric_data_frame = [current_metric_data_frame_mean_all.rename('mean_All').to_frame(),
                                     current_metric_data_frame_mean['Middle'].rename('mean_Middle').to_frame(),
                                     current_metric_data_frame_mean['Inf/Sup'].rename('mean_Inf/Sup').to_frame(),
                                     current_metric_data_frame_std_all.rename('std_All').to_frame(),
                                     current_metric_data_frame_std['Middle'].rename('std_Middle').to_frame(),
                                     current_metric_data_frame_std['Inf/Sup'].rename('std_Inf/Sup').to_frame()]

        excel_data_frame[current_metric] = pd.concat(current_metric_data_frame, axis=1)

    # save
    final_df = pd.concat({k: v.T for k, v in excel_data_frame.items()}, axis=0)
    with pd.ExcelWriter(os.path.join(output_folder, f'{prefix}_performance_result.xlsx'),
                        engine='xlsxwriter') as writer:
        for idx in final_df.index.get_level_values(0).unique():
            temp = final_df.loc[idx]
            temp.to_excel(writer, sheet_name=idx)

    return excel_data_frame


if __name__ == '__main__':
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # parameter settings
    data_root = r'D:\backup\ForMuhan\ClinicalAIModel_PeformanceAnalysis\result_500\extracted_data'
    output_folder = r'result_test_500'
    metrics_result_name = r'metrics_result.pkl'

    is_step_1 = False
    is_step_2_3 = True

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    metrics_result = os.path.join(output_folder, metrics_result_name)

    # step 1: calculate the metrics for all the data
    if is_step_1:
        logger.info('Calculating the metrics for all the cases ...')
        metrics_result = dataset_result_generation(data_path=data_root,
                                                   output_path=os.path.join(output_folder, metrics_result_name))

    if is_step_2_3:
        # step 2: analyze the metrics result, calculate the statistics
        logger.info('Analyzing the resultant metrics, and calculating the statistics ...')
        dataset_performance_result = dataset_performance_analysis(result=metrics_result)

        # step 3: visualization and generate excel report
        logger.info('Visualizing the results and saving the results into Excel ...')

        result_visualization_and_excel_generation(result=dataset_performance_result,
                                                  output_folder=output_folder)

    print('Congrats! May the force be with you ...')
