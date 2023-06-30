#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for extracting the activation vectors. The model must be trained prior to
running this script.
"""

import os

import numpy as np
import h5py
import torch
import torchvision
from tqdm import tqdm

from cifar_extract import load_model
from common_args import TrainExtractParser

def get_label_activations(label_file_path, layer_name):
    with h5py.File(label_file_path, 'r') as f:
        label_activations = f[layer_name][:]
    
    # equivalent to torch.nn.functional.relU
    label_activations = np.maximum(0, label_activations)

    if label_activations.ndim == 2:
        return label_activations

    if label_activations.ndim != 4:
        raise ValueError(f'Unsupported label_activations shape: {label_activations.shape}')
    
    assert label_activations.shape[2] == label_activations.shape[3]

    (_, m, _, _) = label_activations.shape

    # equivalent to np.vstack([layer_activations_i[:, :, j, k] in product(range(num_patches), range(num_patches))])
    label_activations = np.transpose(label_activations, (2, 3, 0, 1))
    return np.reshape(label_activations, (-1, m))

if __name__ == '__main__':
    parser = TrainExtractParser(
        description = 'Extract full activation vectors.',
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
    output_dir = cache_dir / 'activations' / f'{dataset}_ResNet18_Custom_Aug' / 'full_activations'
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

    print('Collecting activations...')
    
    layer_names = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and not 'shortcut' in name:
            layer_names.append(name)

    os.makedirs(output_dir, exist_ok=True)

    for layer_name in layer_names:
        print(f'for layer {layer_name}')

        meta = []
        r, c = 0, -1

        meta_pth = output_dir / (f'meta_{layer_name}.csv')
        print(f'saving layer activation meta data to {meta_pth}')
        with open(meta_pth, 'w') as mf:
            mf.write('label\n')

            # Determine shape of layer_activations and collect meta data
            for i, label_name in enumerate(tqdm(label_names, desc='getting meta data')):
                with h5py.File(activation_dir / f'label{i}.hdf5', 'r') as f:
                    assert f[layer_name].ndim == 4

                    (n, m, num_patches_n, num_patches_m) = f[layer_name].shape

                    assert num_patches_n == num_patches_m

                    if c == -1:
                        c = m
                    else:
                        assert c == m
                    
                    _r = n * num_patches_n ** 2
                    r += _r

                    for _ in range(_r):
                        mf.write(label_name)
                        mf.write('\n')

        act_pth = output_dir / (f'{layer_name}.npy')
        print(f'saving layer activations to {act_pth}')
        layer_activations = np.lib.format.open_memmap(
            filename=act_pth,
            dtype=float,
            mode='w+',
            shape=(r, c)
        )

        at = 0
        for i, label_name in enumerate(tqdm(label_names, desc='getting activations')):
            label_activations = get_label_activations(activation_dir / f'label{i}.hdf5', layer_name)

            (n, _) = label_activations.shape

            layer_activations[at:at + n, :] = label_activations
            del label_activations

            at += n

        assert at == layer_activations.shape[0]
        layer_activations.flush()
        