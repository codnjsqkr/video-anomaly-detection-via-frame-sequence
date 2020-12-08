import torch
import torch.nn as nn
import torch.nn.functional as F

# 깃헙에 돌아다니는 코드 살짝 변형

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_of_img, dataset):
        super(ResNet, self).__init__()
        self.in_planes = 64

        in_channels = num_of_img
        num_classes = 4*5*2 # 출력 모양이 BCEL랑 다르게 채널 한개가 더 많음 (그래야 argmax 사용 가능하니까)
        self.grid_num = num_of_img

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(39424, num_classes)

        # if dataset == 'ped2': 
        #     self.linear = nn.Linear(39424, num_classes)
        # elif dataset == 'avenue':
        #     self.linear = nn.Linear(112640, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        out = out.view(batch_size, 2, 4, 5) # 출력 모양을 2 X 이미지 개수 X 이미지 개수로 reshape

        return out


def ResNet18(num_of_img, dataset):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_of_img, dataset)


def ResNet34(num_of_img, dataset): # 이건 안해봤는데 아마 작동될듯
    return ResNet(BasicBlock, [3, 4, 6, 3], num_of_img, dataset)


# 아래 모델들은 82번줄 숫자를 수정해야함

# def ResNet50(num_of_img):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_of_img)


# def ResNet101(num_of_img):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_of_img)


# def ResNet152(num_of_img):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_of_img)

if __name__ == "__main__":
    # net = ResNet18(4)

    test = torch.rand(4, 12, 240, 360)

    out = ResNet34(4,test)

    # print(out.shape)
    print(out)