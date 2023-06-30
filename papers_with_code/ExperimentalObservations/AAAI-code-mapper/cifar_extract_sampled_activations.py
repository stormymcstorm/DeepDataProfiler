#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 03:13:25 2021

"""

import os
import sys
from glob import glob

import h5py
import torch
import torchvision
import numpy as np
import pandas as pd

from cifar_extract import load_model, get_activations, read_activation
from common_args import TrainExtractParser

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

    layer_names = []
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and not 'shortcut' in name:
            layer_names.append(name)

    os.makedirs(output_dir, exist_ok=True)

    print('Collecting activations...')

    for layer_name in layer_names:
        print(f'for layer {layer_name}')
        
    # all_files = glob(os.path.join(activation_dir, '*.hdf5'))
    
    # names = ['airplane',
    #   'automobile',
    #   'bird',
    #   'cat',
    #   'deer',
    #   'dog',
    #   'frog',
    #   'horse',
    #   'ship',
    #   'truck']
    
    # layers_name = ['layer4.1.bn2', 'layer4.1.bn1', 'layer4.0.bn2', 'layer4.0.bn1', 'layer3.1.bn2', 'layer3.1.bn1',
    #                'layer3.0.bn2', 'layer3.0.bn1', 'layer2.1.bn2', 'layer2.1.bn1', 'layer2.0.bn2', 'layer2.0.bn1',
    #                'layer1.1.bn2', 'layer1.1.bn1', 'layer1.0.bn2', 'layer1.0.bn1']
        
    
    for idx in range(len(layers_name)):
        print("collection sampled activations for", layers_name[idx])
        
        layer = layers_name[idx]
    
        layer_activations = []
        for i in range(num_classes):
            print(i)
            layer_activations_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), layer)
            num_dims = layer_activations_i.shape[1]
            # predictions_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), "predictions")
            # predictions_i = np.array(['true' if p == i else 'false' for p in predictions_i]).reshape(-1,1)
            label_i = np.repeat(i, len(layer_activations_i)).reshape(-1,1)
            # layer_activations_i = np.hstack([label_i, predictions_i, layer_activations_i])
            layer_activations_i = np.hstack([label_i, layer_activations_i])
            layer_activations.append(layer_activations_i)
        
        layer_activations = np.vstack([layer_activations_i for layer_activations_i in layer_activations])
        layer_activations_df = pd.DataFrame(layer_activations)
    
        # cols = np.array(['label', 'predictions'])
        cols = np.array(['label'])
        cols = np.concatenate((cols,np.arange(1,num_dims+1).astype("str")))
    
        layer_activations_df.columns = cols 
        layer_activations_df['label'] = [names[int(layer_activations_df['label'].iloc[i])] for i in range(len(layer_activations_df))]
        print(layer_activations_df.shape)
        # print(layer_activations_df.iloc[:20, :20])
        
        # print("prediction accuracy:", np.sum(layer_activations_df['predictions']=='true')/len(layer_activations_df['predictions']))
        layer_activations_df.to_csv(output_dir+"train_single_batch_"+layer+".csv", index=False)
    
    
    