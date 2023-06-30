#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 02:41:48 2021

"""

import os
import sys


import numpy as np
import pandas as pd
import torchvision
import torch.nn.functional as F

from mapper_interactive.mapper_CLI import get_mapper_graph
from get_knn import elbow_eps
from common_args import GetMapperParser

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = GetMapperParser(
      description = 'Generate a mapper graph from full activations',
      dataset_choices=['CIFAR10']
    )

    parser.add_argument(
      '--seed', type=int, default=42
    )

    args = parser.parse_args()

    dataset = args.dataset
    layer = args.layer
    interval = args.interval
    overlap = args.overlap
    
    num_classes = 10

    cache_dir = args.cache_dir
    dataset_dir = cache_dir / 'datasets' / dataset
    mapper_dir = cache_dir / 'mapper_graphs' / f'{dataset}_ResNet18_Custom_Aug' / 'full_batches'
    
    activation_pth = cache_dir / 'activations' / f'{dataset}_ResNet18_Custom_Aug' \
      / 'full_activations' / f'{layer}.npy'
    cat_path = cache_dir / 'activations' / f'{dataset}_ResNet18_Custom_Aug' \
      / 'full_activations' / f'meta_{layer}.csv'

    if dataset == 'CIFAR10':
      norm_mean = np.array((0.4914, 0.4822, 0.4465))
      norm_std = np.array((0.2023, 0.1994, 0.2010))
      transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(norm_mean.tolist(),
                                                                                    norm_std.tolist())])

      train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=False,
                                            transform=transforms)
      num_classes = 10

    label_names = train.classes

    rng = np.random.default_rng(args.seed)

    print(f'Loading activations from {activation_pth}')
    layer_activations = np.load(
      file = activation_pth,
      mmap_mode='r',
      allow_pickle=True,
    )

    categorical = pd.read_csv(cat_path)

    if layer_activations.shape[0] > 80_000:
      selected_indices = np.random.choice(layer_activations.shape[0], 80_000, replace=False)
      layer_activations = layer_activations[selected_indices, :]
      categorical = categorical.iloc[selected_indices, :]

    if args.eps is not None:
      eps = args.eps
    else:
      eps = elbow_eps(layer_activations)
    print("eps", eps)
            
    min_samples = 5
    

    os.makedirs(mapper_dir, exist_ok=True)
    mapper_pth = mapper_dir \
      / f'mapper_full_batches_{layer}_{interval}_{overlap}_{eps}.json'
    
    get_mapper_graph(
      layer_activations,
      categorical,
      interval, 
      overlap,
      eps, 
      min_samples, 
      mapper_pth,
      is_parallel=False
    )
    
    

