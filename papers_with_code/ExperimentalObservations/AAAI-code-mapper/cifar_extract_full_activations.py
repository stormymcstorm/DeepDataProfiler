#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for extracting the activation vectors. The model must be trained prior to
running this script.
"""

import os
import sys

import numpy as np
import h5py
import torch
import torchvision

from cifar_train import ResNet18
from cifar_extract import load_model, inv_normalize, get_activations
from common_args import TrainExtractParser

if __name__ == '__main__':
    parser = TrainExtractParser(
        description = 'Extract full activation vectors.',
        batch_size=2048
    )

    args = parser.parse_args()

    dataset = args.dataset.upper()
    batch_size = args.batch_size
    num_workers = args.workers

    cache_dir = args.cache_dir
    dataset_dir = cache_dir / 'datasets' / dataset
    activation_dir = cache_dir / 'activations' / f'{dataset}_ResNet18_Custom_Aug' / 'full_activations'
    model_path = cache_dir / 'models' / f'{dataset}_ResNet18_Custom_Aug' / 'best.pth'

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=False,
                                             transform=transforms)
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=False,
                                            transform=transforms)
        num_classes = 10
    if dataset == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=False,
                                              transform=transforms)
        test = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=False,
                                             transform=transforms)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    net = load_model(model_path, num_classes)
    net.eval()

    os.makedirs(activation_dir, exist_ok=True)

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter, is_sample=False, is_imgs=True)

        # with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
        #     [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
        #      activations.items()]
            
        for layer_name, layer_act in activations.items():
            out_dir = activation_dir / layer_name
            os.makedirs(out_dir, exist_ok=True)

            with h5py.File(out_dir / f'label{label_filter}.hdf5', 'w') as out_file:
                out_file.create_dataset(layer_name, data=layer_act)

        del activations

    