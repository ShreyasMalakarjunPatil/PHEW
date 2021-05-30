import torch
import torch.nn as nn
from Models.Layers.Layers import MaskedLinear, MaskedConv2d
from torch.nn.parameter import Parameter
from torch.nn import init


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            MaskedConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MaskedConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                              bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        width = int(out_channels * (base_width / 64.))
        self.residual_function = nn.Sequential(
            MaskedConv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            MaskedConv2d(width, width, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            MaskedConv2d(width, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                              bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, base_width, num_classes=200, dense_classifier=False):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            MaskedConv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, base_width)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, base_width)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, base_width)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskedLinear(512 * block.expansion, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinear, MaskedConv2d) ):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layer_list = []
        for stride in strides:
            layer_list.append(block(self.in_channels, out_channels, stride, base_width))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layer_list)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1


def _resnet(arch, block, num_block, base_width, num_classes):
    model = ResNet(block, num_block, base_width, num_classes)
    return model


def resnet18(input_shape, num_classes):
    """ return a ResNet 18 object
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], 64, num_classes)


def resnet34(input_shape, num_classes):
    """ return a ResNet 34 object
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], 64, num_classes)


def resnet50(input_shape, num_classes):
    """ return a ResNet 50 object
    """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], 64, num_classes)


def resnet101(input_shape, num_classes):
    """ return a ResNet 101 object
    """
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], 64, num_classes)


def resnet152(input_shape, num_classes):
    """ return a ResNet 152 object
    """
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], 64, num_classes, dense_classifier, pretrained)



