# -*- coding: utf-8 -*-
#

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from PIL import Image

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder


def test_single_img_net(cfg):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    
    # Set up input img
    img1_path = '/EVO970_1TB/huang/image2sketch/PhotoSketch/results/00.png'
    img1_np = cv2.imread(img1_path)
    sample = np.array([img1_np])
    
    # Set up data loader
    # Set up data augmentation
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    rendering_images = test_transforms(rendering_images=sample)
    rendering_images = rendering_images.unsqueeze(0)
    
    with torch.no_grad():
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)

    generated_volume = generated_volume.squeeze(0)

    img_dir= '/EVO970_1TB/huang/sketch3d/Pix2Vox/outputs/'
    gv = generated_volume.cpu().numpy()
    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir), epoch_idx)

   









