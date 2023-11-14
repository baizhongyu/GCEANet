# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:01:58 2020

@author: ZJU
"""

import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import Ls_Lc as net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = []
    # transform_list = [transforms.Resize(size=(256, 256))]
    # transform_list = [transforms.Resize(size=(512, 512))]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/home/zxy/IEContraAST1/data/train2014',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/home/zxy/IEContraAST1/data/archive/train',
                    help='Directory path to a batch of style images')
parser.add_argument('--cs_dir', type=str, default='/home/zxy/IEContraAST1/data/archive/train',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='/home/zxy/IEContraAST1/model/vgg_normalised.pth')
parser.add_argument('--sample_path', type=str, default='samples', help='Derectory to save the intermediate samples')

# training options
parser.add_argument('--save_dir', default='./experiments11',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--contrastive_weight_c', type=float, default=0.3)
parser.add_argument('--contrastive_weight_s', type=float, default=0.3)
parser.add_argument('--gan_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=0)
args = parser.parse_args('')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg

valid = 1
fake = 0
D = net.MultiDiscriminator()
D.to(device)

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
network = net.Net(vgg, decoder, args.start_iter)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()
cs_tf = train_transform()
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)
cs_dataset = FlatFolderDataset(args.cs_dir, cs_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=int(1),
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=int(1),
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
cs_iter = iter(data.DataLoader(
    cs_dataset, batch_size=int(1),
    sampler=InfiniteSamplerWrapper(cs_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([{'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()},
                              {'params': network.proj_style.parameters()},
                              {'params': network.proj_content.parameters()}], lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

if(args.start_iter > 0):
    optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

content_images = next(content_iter).to(device)
style_images = next(style_iter).to(device)
cs_images = next(cs_iter).to(device)
######################################################
# content_images_ = content_images[1:]
# content_images_ = torch.cat([content_images_, content_images[0:1]], 0)
# content_images = torch.cat([content_images, content_images_], 0)
# style_images = torch.cat([style_images, style_images], 0)
    ######################################################

loss_c, loss_s= network(content_images, style_images,cs_images)
print(loss_s)
print(loss_c)
