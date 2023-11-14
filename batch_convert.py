
from __future__ import print_function
import torch

import net#_GCEA as net
import time
import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from os.path import basename
from os.path import splitext
import os
import random
import matplotlib.pyplot as plt
import time
###############################################################################################
def imshow(inp, title=None):
    """Imshow for Tensor."""


    inp = inp.cpu().numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

###############################################################################################
# content_dir = '/home/bzy/SSS_recognition/upload/2-Style_transfer/000-Data/content_gray/'
# style_dir = '/home/bzy/SSS_recognition/upload/2-Style_transfer/000-Data/style/gray/'
# save_dir = '/home/bzy/GCEANet/experiments/ablation/gray/Ours_SNL'
# content_dir = '/home/bzy/SSS_recognition/2-Style_transfer/000-Data/content/'
# style_dir = '/home/bzy/SSS_recognition/upload/2-Style_transfer/000-Data/style/color'
# save_dir = '/home/bzy/GCEANet/experiments/ablation/color/Ours_SNL'

content_dir = './sonar/Content/'
style_dir = '/sonar/Style'
save_dir = './output'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    #transforms.Grayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

Content_set = torchvision.datasets.ImageFolder(content_dir, transform)
Style_set = torchvision.datasets.ImageFolder(style_dir, transform)

###############################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
decoder = net.decoder
transform = net.Transform(in_planes = 512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load('./experiments/decoder_iter_160000.pth'))
transform.load_state_dict(torch.load('./experiments/transformer_iter_160000.pth'))
vgg.load_state_dict(torch.load('model/vgg_normalised.pth'))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)



###############################################################################################################################################
#######     Stylization  ######################################################################################################################
###############################################################################################################################################

style_number = 2

cuda = True
# print('centent_set',Content_set)
# print('style_set',Style_set)
number_of_samples = len(Content_set)
print('total', number_of_samples)
number_of_styles = len(Style_set)
print('style number :', len(Style_set))




with torch.no_grad():
    start_time = time.time()
    for i in range(0, number_of_samples):

        cont_img = Content_set[i][0].unsqueeze(0)
        cont_img = cont_img.to(device)

        cont_dir = Content_set.samples[i][0]
        content_name = splitext(basename(cont_dir))[0]
        #print('content_name', content_name)
        cont_label = Content_set.classes[Content_set[i][1]]
        #print('total', number_of_samples, 'current:', i, ':', cont_dir)
        style_idx = random.sample(range(len(Style_set)), style_number)

        for j in style_idx:
            styl_dir = Style_set.samples[j][0]
            styl_name = splitext(basename(styl_dir))[0]
            #print('styl_name', splitext(basename(styl_name))[0])
            styl_img = Style_set[j][0].unsqueeze(0)
            styl_img = styl_img.to(device)

            for k in range(1,2):

                Content3_1 = enc_3(enc_2(enc_1(cont_img)))
                Content4_1 = enc_4(enc_3(enc_2(enc_1(cont_img))))
                Content5_1 = enc_5(Content4_1)

                Style3_1 = enc_3(enc_2(enc_1(styl_img)))
                Style4_1 = enc_4(enc_3(enc_2(enc_1(styl_img))))
                Style5_1 = enc_5(Style4_1)

                #stylized_img = decoder(transform(Content3_1, Style3_1, Content4_1, Style4_1, Content5_1, Style5_1))
                stylized_img = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
                stylized_img.clamp(0, 255)

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                if not os.path.exists(os.path.join(save_dir, cont_label)):
                    os.mkdir(os.path.join(save_dir, cont_label))
                #### save img
                save_name = os.path.join(save_dir, cont_label, content_name + '-' + styl_name + '.jpg')
                torchvision.utils.save_image(stylized_img.cpu().detach().squeeze(0), save_name)

        img_time = time.time() - start_time
    print('total time cost: ', img_time)
    print('single time cost: ', img_time / 300)
            # grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
            # ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            # out_img = Image.fromarray(ndarr)
            # out_img.save(save_name)
            # out_img.show()



