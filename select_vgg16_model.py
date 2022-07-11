import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE
from torchvision.models.densenet import OrderedDict, _DenseBlock, _Transition

def SelfCorrelationPercPooling(x):
    # 解析输入特征形状
    bsize, nb_feats, nb_rows, nb_cols = x.shape
    nb_maps = nb_rows * nb_cols
    # 自相关
    x_3d = torch.reshape(x, [-1, nb_feats, nb_maps])
    x_3d = F.normalize(x_3d, 1)
    x_3d = x_3d  # .cpu()
    x_corr_3d = torch.matmul(x_3d[0, ...].T, x_3d[0, ...])
    x_corr = torch.reshape(x_corr_3d, [-1, nb_maps, nb_rows, nb_cols])
    # top
    x_sort, _ = torch.topk(x_corr, k=512, dim=1, sorted=True)
    return x_sort  # .cuda()

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

        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, x, x_seg):
        input_size = x.size()[2:]

        stage1 = self.backbone_layer1(x)
        stage1_maxpool = self.maxpool(stage1)

        stage2 = self.backbone_layer2(stage1_maxpool)
        stage2_maxpool = self.maxpool(stage2)

        stage3 = self.backbone_layer3(stage2_maxpool)
        stage3_maxpool = self.maxpool(stage3)
        tmp_size = stage3.size()[2:]

        stage4 = self.backbone_layer4(stage3_maxpool)
        stage4_maxpool = self.maxpool(stage4)

        stage5 = self.backbone_layer5(stage4_maxpool)

        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)
        #######################################

        sconv5 = F.interpolate(stage5, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((sconv5, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)
        #######################################

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        pred_flux = pred_flux.squeeze(1)

        return pred_flux

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

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
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.backbone_layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        #######################################

        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, x, x_seg):
        input_size = x.size()[2:]

        stage1 = self.backbone_layer1(x)
        stage1_maxpool = self.maxpool(stage1)

        stage2 = self.backbone_layer2(stage1_maxpool)
        stage2_maxpool = self.maxpool(stage2)

        stage3 = self.backbone_layer3(stage2_maxpool)
        stage3_maxpool = self.maxpool(stage3)
        tmp_size = stage3.size()[2:]

        stage4 = self.backbone_layer4(stage3_maxpool)
        stage4_maxpool = self.maxpool(stage4)

        stage5 = self.backbone_layer5(stage4_maxpool)

        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)
        #######################################

        sconv5 = F.interpolate(stage5, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((sconv5, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)
        #######################################

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        pred_flux = pred_flux.squeeze(1)

        return pred_flux

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * Bottleneck.expansion, 1000)

        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, x_seg):
        input_size = x.size()[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        tmp_size = x.size()[2:]
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((x, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        pred_flux = pred_flux.squeeze(1)

        return pred_flux

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 23, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * Bottleneck.expansion, 1000)

        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, x_seg):
        input_size = x.size()[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        tmp_size = x.size()[2:]
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((x, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)

        pred_flux = pred_flux.squeeze(1)

        return pred_flux

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()

        conv_block = BasicConv2d
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC
        inception_d = InceptionD
        inception_e = InceptionE

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout()
        # self.fc = nn.Linear(2048, 1000)
        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_seg):
        input_size = x.size()[2:]

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        tmp_size = x.size()[2:]

        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)

        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((x, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)
        pred_flux = pred_flux.squeeze(1)

        return pred_flux

class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        block_config = (12, 24, 16)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        # Each denseblock
        num_features = 64
        block = _DenseBlock(
            num_layers=6,
            num_input_features=num_features,
            bn_size=4,
            growth_rate=32,
            drop_rate=0.,
            memory_efficient=False
        )
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + 6 * 32

        trans = _Transition(num_input_features=num_features,
                            num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=4,
                growth_rate=32,
                drop_rate=0.,
                memory_efficient=False
            )
            if i == 0:
                self.features2 = nn.Sequential(block)
            else:
                self.features2.add_module('denseblock%d' % (i + 2), block)
            num_features = num_features + num_layers * 32
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features2.add_module('transition%d' % (i + 2), trans)
                num_features = num_features // 2
        # # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        #
        # # Linear layer
        # self.classifier = nn.Linear(num_features, 1000)

        # Official init from torch repo.

        #######################################
        self.conv_seg = nn.Sequential(nn.Conv2d(1, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.predict_layer = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 64, kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(64, 1, kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_seg):
        input_size = x.size()[2:]
        x = self.features(x)
        tmp_size = x.size()[2:]

        x = self.features2(x)
        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)

        ####################################### 加入分割模块结果
        x_seg = x_seg / x_seg.max()
        x_seg = x_seg.unsqueeze(dim=0)
        sconv_seg = self.conv_seg(x_seg)
        sconv_seg = F.interpolate(sconv_seg, size=tmp_size, mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=tmp_size, mode='bilinear', align_corners=True)
        sconcat = torch.cat((x, sconv_seg), 1)  # 按列拼接

        ####################################### 自相关匹配
        match = SelfCorrelationPercPooling(sconcat)

        pred_flux = self.predict_layer(match)
        pred_flux = F.interpolate(pred_flux, size=input_size, mode='bilinear', align_corners=True)
        pred_flux = pred_flux.squeeze(1)

        return pred_flux
