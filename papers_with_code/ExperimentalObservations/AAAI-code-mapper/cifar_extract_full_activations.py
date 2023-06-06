#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for extracting the full activation vectors. The model must be trained 
prior to running this script. See usage in
```
python cifar_extract_full_activations.py --help
```
for more information on parameters
"""

import argparse
import os
import pathlib


import numpy as np
import h5py
import torch
import torchvision
from cifar_train import ResNet18
from cifar_extract import load_model, inv_normalize, get_activations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract full activation vectors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', choices=['CIFAR10', 'CIFAR100'], type=str.upper, default='CIFAR10',
        help='The dataset to train ResNet18 on.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='The batch size to train with.'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.'
    )
    parser.add_argument(
        '--dataset-dir', type=pathlib.Path, default='datasets',
        help='The directory to read/download datasets from/to.'
    )
    parser.add_argument(
        '--model-dir', type=pathlib.Path, default='models',
        help='The directory to write the trained model to.'
    )
    parser.add_argument(
        '--act-dir', type=pathlib.Path, default='activations',
        help='The directory to write the activations to.'
    )

    args = parser.parse_args()


    # DATASET = sys.argv[1]
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.workers

    norm_mean = np.array((0.4914, 0.4822, 0.4465))
    norm_std = np.array((0.2023, 0.1994, 0.2010))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                  norm_std.tolist())])

    dataset_dir = args.dataset_dir / DATASET
    model_dir = args.model_dir / f'{DATASET}_ResNet18_Custom_Aug'
    activation_dir = args.act_dir / f'{DATASET}_ResNet18_Custom_Aug' / 'full_activations'

    os.makedirs(activation_dir, exist_ok=True)

    if DATASET == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=False,
                                             transform=transforms)
        test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=False,
                                            transform=transforms)
        num_classes = 10
    if DATASET == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=False,
                                              transform=transforms)
        test = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=False,
                                             transform=transforms)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    net = load_model(model_dir / 'best.pth', num_classes)
    net.eval()

    for label_filter in range(num_classes):
        activations = get_activations(net, trainloader, label_filter, is_sample=False, is_imgs=True)
        with h5py.File(os.path.join(activation_dir, f'label{label_filter}.hdf5'), 'w') as out_file:
            [out_file.create_dataset(layer_name, data=layer_act) for layer_name, layer_act in
             activations.items()]
        del activations

    