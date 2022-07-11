import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.backbone_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))
        self.backbone_layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))
        self.backbone_layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))
        self.backbone_layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))
        self.backbone_layer5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        #######################################

        self.predict_layer = nn.Sequential(nn.Conv2d(128, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1),
                                           )


    def SelfCorrelationPercPooling(self, x):
        # 解析输入特征形状
        bsize, nb_feats, nb_rows, nb_cols = x.shape
        nb_maps = nb_rows * nb_cols
        # 自相关
        x = F.normalize(x)
        x_3d = torch.reshape(x, [-1, nb_feats, nb_maps])
        x_corr_3d = torch.Tensor(bsize, nb_maps,nb_maps).cuda()
        for i in range(bsize):
            x_corr_3d[i, ...] = torch.matmul(x_3d[i, ...].T, x_3d[i, ...])
        x_corr = torch.reshape(x_corr_3d, [-1, nb_rows, nb_cols, nb_maps])
        # 映射
        # ranks = torch.round(torch.linspace(1., nb_maps - 1, steps=256)).long().cuda()
        # ranks = ranks.unsqueeze(0).unsqueeze(0).unsqueeze(0)\
        #     .expand(bsize, nb_rows, nb_cols, 256)
        x_sort, _ = torch.topk(x_corr, k=129, dim=-1, sorted=True)
        x_sort = x_sort[:, :, :, 1:]
        # 选出前k个
        # x_sort = torch.gather(x_sort, dim=-1, index=ranks)
        x_sort = x_sort.permute([0, 3, 1, 2])
        return x_sort

    def forward(self, x, x_seg):

        input_size = x.size()[2:]

        stage1 = self.backbone_layer1(x)
        stage1_maxpool = self.maxpool(stage1)

        stage2 = self.backbone_layer2(stage1_maxpool)
        stage2_maxpool = self.maxpool(stage2)

        stage3 = self.backbone_layer3(stage2_maxpool)
        stage3_maxpool = self.maxpool(stage3)

        stage4 = self.backbone_layer4(stage3_maxpool)
        stage4_maxpool = self.maxpool(stage4)

        stage5 = self.backbone_layer5(stage4_maxpool)

        tmp_size = stage5.size()[2:]

        ####################################### 加入分割模块结果
        x_seg = F.normalize(x_seg).unsqueeze(dim=1)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((stage5, sconv_seg), 1) #按列拼接

        ####################################### 自相关匹配
        sconcat = self.SelfCorrelationPercPooling(sconcat)

        pred_flux = self.predict_layer(sconcat)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        pred = pred_flux.squeeze(1) #

        return pred


class Post(nn.Module):
    def __init__(self):
        super(Post, self).__init__()

        self.conv_x = nn.Sequential(nn.Conv2d(1, 128, kernel_size=1), nn.ReLU(inplace=True))
        self.conv_norm = nn.Sequential(nn.Conv2d(1, 128, kernel_size=1), nn.ReLU(inplace=True))

        self.conv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 1, kernel_size=1),
                                  )

    def forward(self, x, x_norm):

        x = self.conv_x(x.unsqueeze(dim=1))
        x_norm = self.conv_norm(x_norm.unsqueeze(dim=1))

        x = torch.cat((x, x_norm), 1)
        x = self.conv(x)

        x = x.squeeze(1)

        return x
