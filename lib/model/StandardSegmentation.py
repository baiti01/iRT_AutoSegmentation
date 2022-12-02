#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/21/2020 7:02 PM

# sys
from easydict import EasyDict as edict
import numpy as np

# monai
from monai.metrics import compute_meandice
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete

# torch
import torch
from torch.cuda.amp import GradScaler, autocast

# project
from lib.model.base_model import BaseModel
from lib.model.module.networks import define_generator
from lib.utils.utils import AverageMeter


class StandardSegmentation(BaseModel):
    def __init__(self,
                 optimizer_option,
                 criterion_option,
                 scheduler_option,
                 logger,
                 cfg=None,
                 is_train=True
                 ):
        super(StandardSegmentation, self).__init__(logger, is_train=is_train)

        self.generator = define_generator(cfg.MODEL.GENERATOR)
        self._create_optimize_engine(optimizer_option, criterion_option, scheduler_option)
        self.cfg = cfg
        self.amp = cfg.AUTOMATIC_MIXED_PRECISION
        if self.amp:
            self.scaler = GradScaler()

        self.is_ce = cfg.CRITERION.IS_CE
        if self.is_ce:
            self.dice_weight = cfg.CRITERION.DICE_WEIGHT
            self.ce_weight = cfg.CRITERION.CE_WEIGHT
            self.ce = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
            self.asdiscrete = AsDiscrete(argmax=True, to_onehot=cfg.DATASET.NUM_CLASSES)
        else:
            self.asdiscrete = AsDiscrete(threshold=0.0)

    def set_dataset(self, input):
        if hasattr(self, 'target'):
            del self.target

        if isinstance(input['data'], list):
            self.input = [x['image'] for x in input['data']]
            self.input = torch.cat(self.input, dim=0)
            self.target = [x['label'] for x in input['data']]
            self.target = torch.cat(self.target, dim=0)
        else:
            self.input = input['data']['image']
            self.target = input['data']['label']

        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.target = self.target.cuda()

        self.unique_idx = torch.unique(self.target).long()
        self.target = one_hot(self.target, num_classes=self.cfg.MODEL.GENERATOR.OUTPUT_CHANNELS)
        self.target.requires_grad = False
        if len(self.unique_idx) == 1:
            return -1
        return 0

    def forward(self):
        if self.amp:
            with autocast():
                self.output = self.generator(self.input)
        else:
            self.output = self.generator(self.input)

        del self.input

    def loss_calculation(self):
        class_spatial_mask = torch.zeros_like(self.target)
        class_spatial_mask[:, self.unique_idx] = 1

        if self.amp:
            with autocast():
                self.loss = self.criterion_pixel_wise_loss(self.output, self.target, class_spatial_mask, reduce_axis=[-3, -2, -1])
                self.loss = torch.mean(self.loss) * (self.target.shape[1] - 1) / (len(self.unique_idx) - 1)
                if self.is_ce:
                    self.loss += self.ce(self.output, torch.argmax(self.target, dim=1))
        else:
            self.loss = self.criterion_pixel_wise_loss(self.output, self.target, class_spatial_mask, reduce_axis=[-3, -2, -1])
            self.loss = torch.mean(self.loss) * (self.target.shape[1] - 1) / (len(self.unique_idx) - 1)
            if self.is_ce:
                self.loss += self.ce(self.output, torch.argmax(self.target, dim=1))
        return 0

    def optimize_parameters(self):
        self.forward()
        is_success = self.loss_calculation()
        del self.output
        del self.target

        if is_success == -1:
            return -1
        self.optimizer_generator.zero_grad()

        if self.amp:
            self.scaler.scale(self.loss).backward()
            self.scaler.unscale_(self.optimizer_generator)
            torch.nn.utils.clip_grad_norm(self.generator.parameters(), 0.5)
            self.scaler.step(self.optimizer_generator)
            self.scaler.update()
        else:
            self.loss.backward()
            torch.nn.utils.clip_grad_norm(self.generator.parameters(), 0.5)
            self.optimizer_generator.step()
        return 0

    def record_information(self, current_iteration=None, data_loader_size=None, batch_time=None, data_time=None,
                           indicator_dict=None, writer_dict=None, phase='train'):
        writer = writer_dict['writer']
        if phase == 'train':
            self.losses_train.update(self.loss.item())
            indicator_dict['current_iteration'] += 1
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', self.loss.item(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if current_iteration % self.cfg.TRAIN.PRINT_FREQUENCY == 0:
                msg = 'Iteration: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'LR: {LR:.6f}\t' \
                      'Loss {losses.val:.5f} ({losses.avg:.5f})'.format(
                    current_iteration, data_loader_size,
                    batch_time=batch_time,
                    data_time=data_time,
                    LR=self.schedulers[0].get_last_lr()[0],
                    losses=self.losses_train)
        elif phase == 'val':
            if current_iteration == 0:
                self.losses_val = AverageMeter()
                self.DSC = AverageMeter()
            self.losses_val.update(self.loss.item())

            self.output = self.asdiscrete(self.output[0, ...])[None, ...]
            current_DSC = compute_meandice(self.output, self.target.to(self.output.device), include_background=False)
            self.DSC.update(current_DSC.detach().cpu().numpy())

            del current_DSC
            del self.output
            del self.target

            if current_iteration == data_loader_size - 1:
                global_steps = writer_dict['val_global_steps']
                writer.add_scalar('val_loss', self.loss, global_steps)
                writer_dict['val_global_steps'] = global_steps + 1

            if current_iteration % self.cfg.VAL.PRINT_FREQUENCY == 0:
                msg = 'Val: [{0}/{1}]\t' \
                      'Loss {losses.val:.5f} ({losses.avg:.5f})\t' \
                      'Avg DSC {avg_DSC:.3f}\t' \
                      'DSC {DSC_val} ({DSC_avg})'.format(
                    current_iteration, data_loader_size,
                    losses=self.losses_val,
                    avg_DSC=np.nanmean(self.DSC.avg) * 100,
                    DSC_val=[float('{:.3f}'.format(x * 100)) for x in list(self.DSC.val[0])],
                    DSC_avg=[float('{:.3f}'.format(x * 100)) for x in list(self.DSC.avg)]
                )
        else:
            raise ValueError('Unknown operation in information recording!')
        self.logger.info(msg)
        return self.losses_val.avg


def get_model(cfg, logger, is_train=True):
    optimizer_option = edict({'optimizer': cfg.TRAIN.OPTIMIZER,
                              'generator_lr': cfg.TRAIN.GENERATOR.LR,

                              # adam
                              'beta1': cfg.TRAIN.GAMMA1,
                              'beta2': cfg.TRAIN.GAMMA2,

                              # sgd
                              'momentum': cfg.TRAIN.MOMENTUM,
                              'nesterov': cfg.TRAIN.NESTEROV,
                              'weight_decay': cfg.TRAIN.WEIGHT_DECAY})

    criterion_option = edict({
        'pixel_wise_loss_type': cfg.CRITERION.PIXEL_WISE_LOSS_TYPE,
    })

    scheduler_option = edict({'niter_decay': int(cfg.TRAIN.TOTAL_ITERATION),
                              'lr_policy': cfg.TRAIN.LR_POLICY,
                              'lr_decay_iters': cfg.TRAIN.LR_STEP,
                              'poly_exponent': cfg.TRAIN.POLY_LR_POLICY_EXPONENT,
                              'last_iteration': -1})

    model = StandardSegmentation(optimizer_option=optimizer_option,
                                 criterion_option=criterion_option,
                                 scheduler_option=scheduler_option,
                                 cfg=cfg,
                                 is_train=is_train,
                                 logger=logger)

    return model
