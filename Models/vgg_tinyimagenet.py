import math
import torch.nn as nn
from Models.Layers.Layers import MaskedLinear, MaskedConv2d
import torch

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
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
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.layers = nn.Sequential(*layers)
        dim = 512 * 4

        self.classifier = nn.Sequential(
            MaskedLinear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MaskedLinear(dim // 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            MaskedLinear(dim // 2, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                layer_size = weight_mask[i].size()
                if len(layer_size)>2:
                    standard_deviation = torch.sqrt(torch.numel(weight_mask[i])*1.0/(layer_size[0]*layer_size[2]*layer_size[3]*torch.sum(weight_mask[i]))).data
                else:
                    standard_deviation = torch.sqrt(torch.numel(weight_mask[i])*1.0/(layer_size[0]*torch.sum(weight_mask[i]))).data
                nn.init.normal_(m.weight, mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_evci(self, weight_mask):
        print('Kaiming Evci Initialization')
        i = 0
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                num_neurons = weight_mask[i].size()[0]
                if len(weight_mask[i].size()) > 2:
                    kernel_size = weight_mask[i].size()[2]*weight_mask[i].size()[3]
                else:
                    kernel_size = 1.0
                for j in range(len(m.weight)):
                    if torch.sum(weight_mask[i][j]) == 0:
                        standard_deviation = 0.000000000000001
                    else:
                        standard_deviation = torch.sqrt(torch.numel(weight_mask[i][j])*1.0/(num_neurons*kernel_size*torch.sum(weight_mask[i][j]))).data
                    nn.init.normal_(m.weight[j], mean=0.0, std=standard_deviation)
                i = i + 1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
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