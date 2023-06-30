#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for extracting the activations for each label. The model must be trained prior to
running this script.
"""

import os

import numpy as np
import h5py
import torch
import torchvision
from tqdm import tqdm

from cifar_extract import load_model, get_activations
from common_args import TrainExtractParser

if __name__ == '__main__':
    parser = TrainExtractParser(
        description = 'Extract label activations',
        batch_size=2048
    )

    parser.add_argument('--skip', action='store_true')

    args = parser.parse_args()

    dataset = args.dataset.upper()
    batch_size = args.batch_size
    num_workers = args.workers

    cache_dir = args.cache_dir
    dataset_dir = cache_dir / 'datasets' / dataset
    activation_dir = cache_dir / 'activations' / f'{dataset}_ResNet18_Custom_Aug'
    model_path = cache_dir / 'models' / f'{dataset}_ResNet18_Custom_Aug' / 'best.pth'

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=False,
                                             transform=transforms)
    if dataset == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=False,
                                              transform=transforms)
    
    label_names = train.classes
    num_classes = len(label_names)

    net = load_model(model_path, num_classes)
    net.eval()

    trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    os.makedirs(activation_dir, exist_ok=True)

    print('Measuring activations...')

    for label_filter, label_name in enumerate(label_names):
        print(f'measuring activations for label {label_filter} ({label_name})')
        activations = get_activations(net, trainloader, label_filter, is_sample=False, is_imgs=True)

        with h5py.File(activation_dir / f'label{label_filter}.hdf5', 'w') as out_file:
            for layer_name, layer_act in activations.items():
                out_file.create_dataset(layer_name, data=layer_act)

        del activations
        