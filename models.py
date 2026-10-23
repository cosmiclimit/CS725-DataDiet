"""Models for 32x32 CIFAR images.

  resnet18 : the target model (and the "expensive" scorer), He et al. 2016, CIFAR version:
             first layer is a 3x3 convolution with stride 1 and there is no max-pool,
             so small images are not shrunk too early. About 11 M weights.
  tiny     : the cheap scorer, three conv blocks (32-64-128 channels). About 0.1 M weights
             and roughly 50x fewer multiply-adds per image than ResNet-18.
  resnet50 : optional bigger target for stretch experiments (not used by default).
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convolutions plus a shortcut: out = relu(F(x) + x)."""
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:            # shapes differ: project x with a 1x1 conv
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    """1x1 reduce, 3x3, 1x1 expand (x4), plus a shortcut. Used by ResNet-50."""
    expansion = 4

    def __init__(self, cin, width, stride=1):
        super().__init__()
        cout = width * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, blocks_per_stage, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for width, n, stride in zip([64, 128, 256, 512], blocks_per_stage, [1, 2, 2, 2]):
            for b in range(n):
                layers.append(block(cin, width, stride if b == 0 else 1))
                cin = width * block.expansion
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)                          # logits (no softmax)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def TinyCNN(num_classes=10):
    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128),          # 32 -> 16 -> 8 -> 4
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))


MODELS = {"resnet18": ResNet18, "tiny": TinyCNN, "resnet50": ResNet50}


def make_model(name, num_classes):
    return MODELS[name](num_classes)
