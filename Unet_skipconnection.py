'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out




class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.dilate1(x)
        out = self.dilate2(out)
        out = self.dilate3(out)
        out = self.conv1x1(out)
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        print('x.size(1)', x.size(1))
        print('x.size(2)', x.size(2))
        print('x.size(3)', x.size(3))
        print('self.layer1', self.layer1.shape)
        print('self.layer2', self.layer2.shape)
        print('self.layer3', self.layer3.shape)
        print('self.layer4', self.layer4.shape)
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)


        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        base_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.EnDown1 = EnDown(in_channels=256, out_channels=256 * 2)
        self.dblock1 = DACblock(channel=128 * 2)  # Update the argument name
        self.spp1 = SPPblock(in_channels=128 * 2)  # Update the argument name
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.EnDown2 = EnDown(in_channels=base_channels * 2, out_channels=base_channels * 4)
        self.dblock2 = DACblock(channel=base_channels * 4)  # Update the argument name
        self.spp2 = SPPblock(in_channels=base_channels * 4)  # Update the argument name
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.EnDown3 = EnDown(in_channels=base_channels * 4, out_channels=base_channels * 8)
        self.dblock3 = DACblock(channel=base_channels * 8)  # Update the argument name
        self.spp3 = SPPblock(in_channels=base_channels * 8)  # Update the argument name
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dblock4 = DACblock(channel=base_channels * 8)  # Update the argument name
        self.spp4 = SPPblock(in_channels=base_channels * 8)  # Update the argument name
        self.linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out1_1 = self.layer1(out)
        out1_2 = self.EnDown1(out1_1)
        out1_2 = self.dblock1(out1_1)
        out1_2 = self.spp1(out1_1)
        print(out1_2.shape)
        out2_1 = self.layer2(out1_2)
        out2_2 = self.EnDown2(out2_1)
        out2_2 = self.dblock2(out2_1)
        out2_2 = self.spp2(out2_1)
        out3_1 = self.layer3(out2_2)
        out3_2 = self.EnDown3(out3_1)
        out3_2 = self.dblock3(out3_1)
        out3_2 = self.spp3(out3_1)
        out4_1 = self.layer4(out3_2)
        out4_2 = self.EnBlock4_4(out4_1)
        out4_2 = self.dblock4(out4_1)
        out4_2 = self.spp4(out4_1)
        out = F.avg_pool2d(out4_1, 4)
        out = out.view(out4_1.size(0), -1)
        out = self.linear(out)
        return out1_1, out2_1, out3_1, out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    with torch.no_grad():

        net = PreActResNet50()
        y = net((torch.randn(1, 3, 512, 512)))
        print(y.size())



