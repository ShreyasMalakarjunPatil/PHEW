import torch.nn as nn
from Models.Layers.Layers import MaskedLinear, MaskedConv2d
import torch.nn.functional as F
import torch
import math

class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample=False):
        super(Block, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = MaskedConv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f_out)
        self.conv2 = MaskedConv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f_out)

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                MaskedConv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(f_out)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    def __init__(self, plan, num_classes, dense_classifier):
        super(ResNet, self).__init__()

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = MaskedConv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = MaskedLinear(plan[-1][0], num_classes)
        if dense_classifier:
            self.fc = MaskedLinear(plan[-1][0], num_classes)

        self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, nn.Linear, MaskedConv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_normal(self):
        print('Normal')
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_uniform(self):
        print('Xavier Uniform')
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_liu(self, weight_mask):
        print('Kaiming Liu Initialization')
        i = 0
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d)):
                standard_deviation = torch.sqrt(torch.numel(weight_mask[i])*2.0/(torch.numel(weight_mask[i][0])*torch.sum(weight_mask[i]))).data
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
                for j in range(len(m.weight)):
                    if torch.sum(weight_mask[i][j]) == 0:
                        standard_deviation = 0.000000000000001
                    else:
                        standard_deviation = torch.sqrt(2.0/torch.sum(weight_mask[i][j])).data
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

def _plan(D, W):
    """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

    The ResNet is structured as an initial convolutional layer followed by three "segments"
    and a linear output layer. Each segment consists of D blocks. Each block is two
    convolutional layers surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and each layer in the
    third segment has 64W filters.

    The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
    N is the total number of layers in the network: 2 + 6D.
    The default value of W is 16 if it isn't provided.

    For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
    linear layer, there are 18 convolutional layers in the blocks. That means there are nine
    blocks, meaning there are three blocks per segment. Hence, D = 3.
    The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
    """
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2 * W, D), (4 * W, D)]

    return plan


def _resnet(arch, plan, num_classes, dense_classifier, pretrained):
    model = ResNet(plan, num_classes, dense_classifier)
    return model

def resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 16)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained)

def resnet32(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 16)
    return _resnet('resnet32', plan, num_classes, dense_classifier, pretrained)

def wide_resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 32)
    return _resnet('wide_resnet20', plan, num_classes, dense_classifier, pretrained)