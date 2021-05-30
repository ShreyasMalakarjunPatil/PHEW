import torch.nn as nn
import torch
from Models.Layers.Layers import MaskedLinear, MaskedConv2d
import math

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class  VGG(nn.Module):
    def __init__(self, cfg, num_classes, batch_norm = False):
        super(VGG, self).__init__()
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=True), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.layers = nn.Sequential(*layers)
        self.fc = MaskedLinear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights_normal(self):
        print('Normal')
        for m in self.modules():
            if isinstance(m, (MaskedConv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                nn.init.normal_(m.weight, 0, math.sqrt(0.1))
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_uniform(self):
        print('Xavier Uniform')
        for m in self.modules():
            if isinstance(m, (MaskedConv2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_liu(self, weight_mask):
        print('Kaiming Liu Initialization')
        i = 0
        for m in self.modules():
            if isinstance(m, (MaskedConv2d)):
                standard_deviation = torch.sqrt(torch.numel(weight_mask[i])*2.0/(torch.numel(weight_mask[i][0])*torch.sum(weight_mask[i]))).data
                nn.init.normal_(m.weight, mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                standard_deviation = torch.sqrt(torch.numel(weight_mask[i]) * 2.0 / (
                            torch.numel(weight_mask[i][0]) * torch.sum(weight_mask[i]))).data
                nn.init.normal_(m.weight, mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights_evci(self, weight_mask):
        print('Kaiming Evci Initialization')
        i = 0
        for m in self.modules():
            if isinstance(m, (MaskedConv2d)):
                for j in range(len(m.weight)):
                    if torch.sum(weight_mask[i][j]) == 0:
                        standard_deviation = 0.000000000000001
                    else:
                        standard_deviation = torch.sqrt(2.0 / torch.sum(weight_mask[i][j])).data
                    nn.init.normal_(m.weight[j], mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                for j in range(len(m.weight)):
                    if torch.sum(weight_mask[i][j]) == 0:
                        standard_deviation = 0.000000000000001
                    else:
                        standard_deviation = torch.sqrt(2.0 / torch.sum(weight_mask[i][j])).data
                    nn.init.normal_(m.weight[j], mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i], bias_mask[i])
                i = i + 1

def vgg11(input_shape, num_classes):
    return VGG(cfg['A'], num_classes)

def vgg11_bn(input_shape, num_classes):
    return VGG(cfg['A'], num_classes, batch_norm = True)

def vgg13(input_shape, num_classes):
    return VGG(cfg['B'], num_classes)

def vgg13_bn(input_shape, num_classes):
    return VGG(cfg['B'], num_classes, batch_norm = True)

def vgg16(input_shape, num_classes):
    return VGG(cfg['D'], num_classes)

def vgg16_bn(input_shape, num_classes):
    return VGG(cfg['D'], num_classes, batch_norm = True)

def vgg19(input_shape, num_classes):
    return VGG(cfg['E'], num_classes)

def vgg19_bn(input_shape, num_classes):
    return VGG(cfg['E'], num_classes, batch_norm = True)