import torch
import torch.nn as nn
from torch.nn import init

from CBAM import CBAMBlock
from resnet import resnet50, resnet18
import torch.nn.functional as F
from SNA import SNA
from fsa_methods import *
from plm import PLM
import pywt

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self, class_num, dataset, arch='resnet50', part_num=10):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset
        self.part_num = part_num

        pool_dim = 2048
        # self.MFGM = DCT_MFGM(1024)

        self.bottleneck_s = nn.BatchNorm1d(pool_dim * (self.part_num + 1))
        self.bottleneck_s.bias.requires_grad_(False)  # no shift
        self.bottleneck_s.apply(weights_init_kaiming)
        self.classifier_s = nn.Linear(pool_dim * (self.part_num + 1), class_num, bias=False)
        self.classifier_s.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # 应用高频扰动
        x = self.low_freq_perturb(x)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x = self.base_resnet.base.layer2(x_)
        x = self.base_resnet.base.layer3(x_)
        # print(x_.shape)
        # print(x_.shape)
        x = self.base_resnet.base.layer4(x_)  # [b,c,h,w]
        part_feat, sim = self.block(x)

        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))  # global feat

        feats = torch.cat((part_feat, x_pool), dim=1)  # global feat and local feat ( concatenate)，

        feat = self.bottleneck(x_pool)

        if self.training:

            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))

            return x_pool, self.classifier(feat), loss_ort, feats, self.classifier_s(self.bottleneck_s(feats))
        else:
            return self.l2norm(feat), self.l2norm(self.bottleneck_s(feats))
            # return self.l2norm(feats), self.l2norm(self.bottleneck_s(feats))

