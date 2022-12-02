#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 8/12/2019 3:43 PM

# sys
import os
import random
import time

# torch and monai
import torch
from monai.inferers import sliding_window_inference

# project
from lib.utils.utils import AverageMeter


def do_validate(val_loader,
                model,
                cfg,
                visualize,
                writer_dict,
                final_output_dir):
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    selected_visualized_data = random.randint(0, len(val_loader) - 1)
    for i, current_data in enumerate(val_loader):
        is_success = model.set_dataset(current_data)
        if is_success == -1:
            continue
        del current_data

        model.input.require_grad = False

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=model.amp):
                model.output = sliding_window_inference(model.input,
                                                        roi_size=cfg.DATASET.TARGET_SIZE,
                                                        sw_batch_size=cfg.VAL.SLIDING_WINDOW_BATCH_SIZE,
                                                        predictor=model.generator,
                                                        mode='gaussian',
                                                        overlap=cfg.VAL.OVERLAP_RATIO,
                                                        device='cpu')
                del model.input

                class_spatial_mask = torch.zeros_like(model.target)
                class_spatial_mask[:, model.unique_idx] = 1
                model.loss = model.criterion_pixel_wise_loss(model.output.cpu(), model.target.cpu())
                model.loss = torch.mean(model.loss) * (model.target.shape[1] - 1) / (len(model.unique_idx) - 1)

                del class_spatial_mask

        batch_time.update(time.time() - end)
        end = time.time()

        performance = model.record_information(current_iteration=i,
                                               data_loader_size=len(val_loader),
                                               writer_dict=writer_dict,
                                               phase='val')

        if i == selected_visualized_data and cfg.IS_VISUALIZE and False:
            visualize(model,
                      writer_dict['val_global_steps'],
                      os.path.join(final_output_dir, "val"),
                      1
                      )

    return performance
