import argparse
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html
# from torchvision.utils import save_image
import ntpath
from PIL import Image
from util import util
import torch
import math
import cv2
import numpy as np
from pathlib import Path
import math
import yaml
import json
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def save_images(visuals, img_path):
    visuals =  util.tensor2im(visuals)
    name = ntpath.basename(img_path[0])
    img = Image.fromarray(visuals)
    img_dir="myresult\outdoor_outdoor_latest"
    img.save(os.path.join(img_dir, name))
    # save_image(visuals, os.path.join(img_dir, name))


def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top,padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top,padding_top + image.size(2)

def dehaze():
    # import os
    import sys
    sys.argv = ['mydetect.py', '--netG', 'net2', '--preprocess', 'None', '--dataroot', './datasets/outdoor', '--name', 'net2_outdoor', '--model', 'net', '--phase', 'test']
    opt = TestOptions().parse()  # get test options
    opt.no_dropout = True
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    opt.psnr = True
    to = ntpath.basename(opt.dataroot) + '_' + opt.name.split('_', 1)[-1]+ '_' + opt.epoch
    img_dir = os.path.join('myresult',to)
    gt_dir = os.path.join(opt.dataroot, 'testB')
    print(img_dir)
    print(gt_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    model.eval()
    for i, data in enumerate(dataset):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        h, w = data['A'].shape[2], data['A'].shape[3]
        # max_h = int(math.ceil(h / 4)) * 4
        # max_w = int(math.ceil(w / 4)) * 4
        max_h = int(math.ceil(h / 128)) * 128
        max_w = int(math.ceil(w / 128)) * 128
        # print(data['A_paths'])
        # print(data['B_paths'])
        data['A'], ori_left, ori_right, ori_top, ori_down = padding_image(data['A'], max_h, max_w)

        data['B'], ori_left, ori_right, ori_top, ori_down = padding_image(data['B'], max_h, max_w)

        print(max_h)
        print(max_w)

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        save_images(visuals['fake_B'][:, :, ori_top:ori_down, ori_left:ori_right], img_path)
        print(i)
        # if i % 5 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))

        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    if opt.psnr:
        from psnr_dzt import psnr
        psnr(img_dir, gt_dir)

if __name__ == '__main__':
    dehaze()