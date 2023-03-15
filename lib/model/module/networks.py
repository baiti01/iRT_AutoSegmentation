#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 2:27 PM

# torch
import torch


def define_generator(generator_config):
    generator_name = generator_config.NAME
    if 'unet_monai' == generator_name.lower():
        from monai.networks.nets import UNet
        network = UNet(dimensions=generator_config.DIMENSIONS,
                       in_channels=generator_config.INPUT_CHANNELS,
                       out_channels=generator_config.OUTPUT_CHANNELS,
                       channels=generator_config.CHANNELS,
                       strides=generator_config.STRIDES,
                       num_res_units=generator_config.NUM_RES_UNITS)
    elif 'segresnet_monai' == generator_name.lower():
        from monai.networks.nets import SegResNet
        network = SegResNet(spatial_dims=generator_config.DIMENSIONS,
                            init_filters=generator_config.INITIAL_FILTERS,
                            out_channels=generator_config.OUTPUT_CHANNELS,
                            norm=generator_config.NORM,
                            blocks_up=generator_config.BLOCKS_UP,
                            blocks_down=generator_config.BLOCKS_DOWN,
                            upsample_mode=generator_config.UPSAMPLE_MODE)
    elif 'lightweight_UNet'.lower() == generator_name.lower():
        from lib.model.module.Lightweight_Unet import UnetGenerator
        network = UnetGenerator(input_channels=generator_config.INPUT_CHANNELS,
                                output_channels=generator_config.OUTPUT_CHANNELS,
                                downsampling_number=generator_config.DOWNSAMPLING_NUMBER,
                                filter_number_last_conv_layer=generator_config.FILTER_NUMBER_LAST_CONV_LAYER,
                                norm_layer=generator_config.NORM_LAYER,
                                dimension=generator_config.DIMENSION)
    else:
        raise ('Unsupported network: {}'.format(generator_name))

    if hasattr(generator_config, 'DROPOUT_BLOCK') and generator_config.DROPOUT_BLOCK.IS_ENABLED:
        from lib.model.module.modules import add_dropblock_layer
        add_dropblock_layer(network,
                            block_size=generator_config.DROPOUT_BLOCK.DROP_BLOCK_SIZE,
                            drop_prob=generator_config.DROPOUT_BLOCK.DROP_PROB)
    network = network.to('cuda' if torch.cuda.is_available() else 'cpu')
    return network


def def_discriminator(discriminator_config):
    raise NotImplementedError('Not implemented yet!')


if __name__ == '__main__':
    from monai.networks.nets import SwinUNETR

    model = SwinUNETR(img_size=(96, 96), in_channels=1, out_channels=1, feature_size=48, spatial_dims=2)
    model = model.to('cuda')
    input = torch.randn((1, 1, 96, 96)).to('cuda')
    output = model(input)

    print('Congrats! May the force be with you ...')
