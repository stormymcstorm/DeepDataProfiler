#! /usr/bin/env python

"""
Script used for training ResNet18 on CIFAR10/CIFAR100.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import collections
import argparse
import os

import torch
import torchvision
import torchvision.transforms as tv_transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback

from common_args import BaseParser

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4, stride=1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def lr_function(epoch):
    if 0 <= epoch < 150:
        return 0.1
    elif 150 <= epoch < 250:
        return 0.01
    else:
        return 0.001


if __name__ == '__main__':
    parser = BaseParser(description = 'Train a ResNet18 model on CIFAR10/CIFAR100 using Catalyst')
    parser.add_dataset_arg()
    parser.add_argument(
        '--workers', type=int, default=4,
        help='How many subprocesses to use for data loading. 0 means the data will be loaded in the main process.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='The batch size to train with.'
    )
    parser.add_argument(
        '--epochs', type=int, default=350,
        help='The number of epochs to run.'
    )

    args = parser.parse_args()

    dataset = args.dataset.upper()
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.workers

    cache_dir = args.cache_dir
    log_dir = cache_dir / 'logs' / f'{dataset}_ResNet18_Custom_Aug'
    dataset_dir = cache_dir / 'datasets' / dataset
    model_dir = cache_dir / 'models' / f'{dataset}_ResNet18_Custom_Aug'


    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    transform_train = tv_transforms.Compose([
        tv_transforms.RandomCrop(32, padding=4),
        tv_transforms.RandomHorizontalFlip(),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    loaders = collections.OrderedDict()

    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True,
                                             transform=transform_train)
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True,
                                            transform=transform_test)
        num_classes = 10
    elif dataset == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True,
                                             transform=transform_train)
        test = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True,
                                            transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(
        train, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    loaders["train"] = trainloader
    loaders["valid"] = testloader

    # model, criterion, optimizer
    net = ResNet18(num_classes)
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda() # TODO: probably remove .cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=log_dir,
        num_epochs=num_epochs,
        callbacks=[AccuracyCallback(num_classes=num_classes, accuracy_args=[1])],
        verbose=True,
        timeit=True,
        load_best_on_end=True
    )

    # Save the model
    os.makedirs(model_dir, exist_ok=True)
    print(f'Saving model to {model_dir / "best.pth"}')
    torch.save(net.module.state_dict(), model_dir / 'best.pth')
