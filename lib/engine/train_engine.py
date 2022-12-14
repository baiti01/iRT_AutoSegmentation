#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 8/12/2019 3:43 PM

# sys
import logging
import os
import time

# project
from lib.engine.validate_engine import do_validate
from lib.utils.utils import AverageMeter, save_checkpoint


def do_train(train_loader,
             val_loader,
             model,
             indicator_dict,
             cfg,
             writer_dict,
             final_output_dir,
             log_dir,
             visualize):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for i, current_data in enumerate(train_loader, start=indicator_dict['current_iteration']):
        data_time.update(time.time() - end)

        if i > indicator_dict['total_iteration']:
            return

        # validation
        if indicator_dict['current_iteration'] % cfg.VAL.EVALUATION_FREQUENCY == 0:
            if False:
                indicator_dict['current_performance'] = do_validate(val_loader, model, cfg, visualize, writer_dict,
                                                                    final_output_dir)
            indicator_dict['is_best'] = False
            if indicator_dict['current_performance'] > indicator_dict['best_performance']:
                indicator_dict['best_performance'] = indicator_dict['current_performance']
                indicator_dict['is_best'] = True

            # save checkpoint
            output_dictionary = {'indicator_dict': indicator_dict,
                                 'writer_dict_train_global_steps': writer_dict['train_global_steps'],
                                 'writer_dict_val_global_steps': writer_dict['val_global_steps'],
                                 'tb_log_dir': log_dir}

            if hasattr(model, 'generator'):
                output_dictionary['generator'] = model.generator.state_dict()
                output_dictionary['optimizer_generator'] = model.optimizer_generator.state_dict()

            save_checkpoint(output_dictionary, indicator_dict, final_output_dir)
            model.train()

        # train
        is_success = model.set_dataset(current_data)
        if is_success == -1:
            model.logger.info('Failed to setup current training sample: {}.'
                              'Go to next one ...'.format(current_data['path'][0]))
            continue

        is_success = model.optimize_parameters()
        if is_success == -1:
            model.logger.info('Failed to update the model parameters: {}. Go to next one ...'.format(current_data['path'][0]))
            continue

        # visualize
        if indicator_dict['current_iteration'] % cfg.TRAIN.DISPLAY_FREQUENCY == 0 and cfg.IS_VISUALIZE and False:
            visualize(model,
                      indicator_dict['current_iteration'],
                      os.path.join(final_output_dir, "train"),
                      cfg.TRAIN.DISPLAY_FREQUENCY)

        # update learning rate
        for current_scheduler in model.schedulers:
            current_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
        model.record_information(i, len(train_loader), batch_time, data_time, indicator_dict, writer_dict,
                                 phase='train')
