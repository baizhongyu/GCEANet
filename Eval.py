import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
import numpy as np
import time

def test_transform():
    transform_list = []
    #transform_list = [transforms.Resize(size=(256, 256))]
    #transform_list = [transforms.Resize(size=(512, 512))]
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = '/home/zxy/桌面/ablation/video_test1/contents/frame_0001.png',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = '/home/zxy/桌面/efficiency_test/2.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'model/vgg_normalised.pth')

# Additional options
parser.add_argument('--decoder', type=str, default = '/home/zxy/MFSANet/experiments/Ours/decoder_iter_180000.pth')
parser.add_argument('--transform', type=str, default = '/home/zxy/MFSANet/experiments/Ours/transformer_iter_180000.pth')
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = '/home/zxy/桌面/efficiency_test',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = net.decoder
transform = net.Transform(in_planes = 512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.load_state_dict(torch.load(args.vgg))

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

content_tf = test_transform()
style_tf = test_transform()

content = content_tf(Image.open(args.content))
style = style_tf(Image.open(args.style))

style = style.to(device).unsqueeze(0)
content = content.to(device).unsqueeze(0)
start_time = time.time()
with torch.no_grad():

    for x in range(args.steps):

        print('iteration ' + str(x))

        Content3_1 = enc_3(enc_2(enc_1(content)))
        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)

        Style3_1 = enc_3(enc_2(enc_1(style)))
        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)

        stylized_img = decoder(transform(Content3_1, Style3_1, Content4_1, Style4_1, Content5_1, Style5_1))

        stylized_img.clamp(0, 255)


    t1 = time.time() - start_time
    print("time: %.5fs" % np.mean(t1))
    print("FPS: %.5fs" % (1 / np.mean(t1)))
    stylized_img = stylized_img.cpu()


    
    output_name = '{:s}/{:s}_OURS_{:s}{:s}'.format(
                args.output, splitext(basename(args.content))[0],
                splitext(basename(args.style))[0], args.save_ext
            )
    save_image(stylized_img, output_name)
    t2 = time.time()-start_time
    print("time: %.5fs" % np.mean(t2))
    print("FPS: %.5fs" % (1/np.mean(t2)))