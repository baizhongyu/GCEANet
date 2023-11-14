# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128)
)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

###########GCEA
class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio=0.25,
                 pooling_type='att',
                 fusion_types=('channel_add',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        ###############################
        self.mk=nn.Linear(self.inplanes, 64,bias=False)
        self.mv=nn.Linear(64, self.inplanes,bias=False)
        self.softmax=nn.Softmax(dim=1)
        ###################################################
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)


        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

            # [N, C, 1, 1]
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                #nn.Linear(self.inplanes, 64, bias=False),
                #nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                #nn.LayerNorm([self.planes, 1, 1]),
                #nn.LayerNorm(64),

                #nn.ReLU(inplace=True),  # yapf: disable

                #nn.Linear(64, self.inplanes, bias=False))
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                #nn.Linear(self.inplanes, 64, bias=False),
                # nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                # nn.LayerNorm([self.planes, 1, 1]),
                #nn.LayerNorm(64),
                #nn.ReLU(inplace=True),  # yapf: disable

                #nn.Linear(64, self.inplanes, bias=False))
                nn.Conv2d(self.inplanes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, Fs, F_s):
        batch, channel, height, width = Fs.size()
        if self.pooling_type == 'att':
            input_Fs = Fs
            # [N, C, H * W]
            input_Fs = input_Fs.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_Fs = input_Fs.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(F_s)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_Fs, context_mask)# [N, 1, C, H * W]# [N, 1, H * W, 1]
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)


        else:
            # [N, C, 1, 1]
            context = self.avg_pool(F_s)

        return context

    def forward(self, content, style):

        Fs = style
        F_s = mean_variance_norm(style)

        # [N, C, 1, 1]
        context = self.spatial_pool(Fs, F_s)
        batch, channel, height, width = content.size()
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            context = content * channel_mul_term

            context = context.view(batch, height * width, channel)
            # [N, H*W,C]

            attn = self.mk(context)  # bs,n,c
            attn = self.softmax(attn)  # bs,n,c
            attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,c
            out = self.mv(attn)  # bs,n,c

            out = out.view(batch, channel, height, width)
            # channel_add_term = self.channel_add_conv(context)
            # #channel_mul_term = channel_add_term.view(batch, channel, height, width)
            # out = channel_add_term.view(batch, channel, height, width)
            #out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            #channel_add_term = self.channel_add_conv(context)
            context = content + context

            context = context.view(batch, height * width, channel)
            # [N, H*W,C]

            attn = self.mk(context)  # bs,n,c
            attn = self.softmax(attn)  # bs,n,c
            attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,c
            out = self.mv(attn)  # bs,n,c
            out = out.view(batch, channel, height, width)
            # context = context.view(batch, -1)
            # [N, C*H*W]
            # channel_add_term = self.channel_add_conv(context)
            # out = channel_add_term.view(batch, channel, height, width)
            #channel_add_term = channel_add_term.view(batch, channel, height, width)
            #out = out + channel_add_term
        return out######



class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = GlobalContextBlock(inplanes=in_planes)
        self.sanet5_1 = GlobalContextBlock(inplanes=in_planes)
        # self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))



class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        # projection
        self.proj_style = projection_style
        self.proj_content = projection_content

        # transform
        self.transform = Transform(in_planes=512)
        self.decoder = decoder
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if (start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False  #####固定特征提取层

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def compute_contrastive_loss(self, feat_q, feat_k, tau, index):
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / tau  ###torch.mm：矩阵相乘
        # print('3',out.size())###([1, 4])
        # loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        loss = self.cross_entropy_loss(out, torch.tensor([index], dtype=torch.long, device=feat_q.device))
        # print('loss',loss)
        return loss

    def style_feature_contrastive(self, input):
        # out = self.enc_style(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_style(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)  ####torch.norm():求p范数
        return out

    def content_feature_contrastive(self, input):
        # out = self.enc_content(input)
        out = torch.sum(input, dim=[2, 3])
        out = self.proj_content(out)
        out = out / torch.norm(out, p=2, dim=1, keepdim=True)
        return out

    def forward(self, content, style, batch_size):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        # print('Relu_3_1', content_feats[2].size())[6, 256, 64, 64]
        # print('Relu_4_1',content_feats[3].size())[6, 512, 32, 32]
        # print('Relu_5_1', content_feats[4].size())[6, 512, 16, 16]
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        #print('stylized',stylized.size())

        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)



        # stylized_c = self.transform(
        #     g_t_feats[3], style_feats[3], g_t_feats[4], style_feats[4])
        # #print('stylized',stylized.size())
        # gt_c = self.decoder(stylized_c)
        # gt_c_feats = self.encode_with_intermediate(gt_c)
        #
        #
        # stylized_s = self.transform(
        #     content_feats[3], g_t_feats[3], content_feats[4], g_t_feats[4])
        # #print('stylized',stylized.size())
        # gt_s = self.decoder(stylized_s)
        # gt_s_feats = self.encode_with_intermediate(gt_s)






        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])

        # Contrastive learning.
        # half = int(batch_size / 2)
        style_new = self.style_feature_contrastive(g_t_feats[2][0:batch_size])
        content_new = self.content_feature_contrastive(g_t_feats[3][0:batch_size])

        style_pos = self.style_feature_contrastive(style_feats[2][0:batch_size])
        content_pos = self.content_feature_contrastive(content_feats[3][0:batch_size])
        #style_pos = self.style_feature_contrastive(gt_s_feats[2][0:batch_size])
        #content_pos = self.content_feature_contrastive(gt_c_feats[3][0:batch_size])
        # style_feats = self.encode_with_intermediate(style)
        # content_feats = self.encode_with_intermediate(content)

        # style_up = self.style_feature_contrastive(g_t_feats[2][0:half])
        # style_down = self.style_feature_contrastive(g_t_feats[2][half:])
        # content_up = self.content_feature_contrastive(g_t_feats[3][0:half])
        # content_down = self.content_feature_contrastive(g_t_feats[3][half:])
        # print('g_t',g_t.size())#####g_t torch.Size([6, 3, 256, 256])
        # print('g_t_feats', g_t_feats.size())

        style_contrastive_loss = 0
        for i in range(int(batch_size)):
            reference_style = style_new[i:i + 1]
            style_comparisons = torch.cat([style_pos[i:i + 1], style_new[0:i], style_new[i + 1:]], 0)
            style_contrastive_loss += self.compute_contrastive_loss(reference_style, style_comparisons, 0.2, 0)

        content_contrastive_loss = 0
        for i in range(int(batch_size)):
            reference_content = content_new[i:i + 1]
            content_comparisons = torch.cat([content_pos[i:i + 1], content_new[0:i], content_new[i + 1:]], 0)
            content_contrastive_loss += self.compute_contrastive_loss(reference_content, content_comparisons, 0.2, 0)




        return g_t, loss_c, loss_s, l_identity1, l_identity2, content_contrastive_loss, style_contrastive_loss