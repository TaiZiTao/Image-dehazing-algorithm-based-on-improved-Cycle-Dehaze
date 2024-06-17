from tensorboardX import SummaryWriter
import torch
# from data import create_dataset
from models import create_model
import sys
from options.test_options import TestOptions
from options.train_options import TrainOptions
sys.argv = ['show_model.py', '--preprocess', 'None', '--gpu_ids', '0', '--dataroot', './datasets/sots', '--name', 'ots_cycledehaze_per', '--model', 'cycle_dehaze_per', '--phase', 'test', '--no_dropout']
opt = TrainOptions().parse()  # get test options
    # hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
# dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)  
input = torch.rand(1, 3, 258, 258).cuda()
print(model.get_net().keys())
with SummaryWriter(comment='lzy') as w:
    w.add_graph(model.get_net()['D_B'], (input,))
# python show_model.py --preprocess None --gpu_ids 0  --dataroot ./datasets/sots --name ots_cycledehaze_per --model cycle_dehaze_per --phase test --no_dropout