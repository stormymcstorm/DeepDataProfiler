#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 02:41:48 2021

"""

import argparse
import os
import sys
import pathlib

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F

from mapper_interactive.mapper_CLI import get_mapper_graph
from get_knn import elbow_eps

def read_activation(filepath, layer):
    with h5py.File(filepath, 'r') as f:
        try:
            activation = f[layer][:]
        except:
            raise ValueError(f'Unrecognized layer {layer}. Please choose from: {list(f.keys())}')
        return activation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get mapper for full activation vectors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'layer', type=str,
        help='The layer to extract activations for'
    )
    parser.add_argument(
        '--interval', type=int, default=10,
        help='Number of hypercubes along each dimension. Sometimes referred to as resolution.'
    )
    parser.add_argument(
        '--overlap', type=int, default=50,
        help='TODO'
    )
    parser.add_argument(
        '--dataset', choices=['CIFAR10'], type=str.upper, default='CIFAR10',
        help='The dataset to train ResNet18 on.'
    )
    parser.add_argument(
        '--act-dir', type=pathlib.Path, default='activations',
        help='The directory to write the activations to.'
    )
    parser.add_argument(
        '--graph-dir', type=pathlib.Path, default='mapper_graphs',
        help='The directory to write the mapper graphs to.'
    )

    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    layer = args.layer
    interval = args.interval
    overlap = args.overlap
    
    DATASET = args.dataset

    if DATASET == 'CIFAR10':
        num_classes = 10
    else:
        raise NotImplementedError(f'Sorry this script does not support {DATASET}')

    activation_dir = args.act_dir / f'{DATASET}_ResNet18_Custom_Aug' / 'full_activations'
    graph_dir = args.graph_dir / 'full_batches'
    
    names = ['airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck']
    
    print("collection full activations for", layer)
    layer_activations = []
    for i in range(num_classes):
        print(i)
        layer_activations_i = read_activation(os.path.join(activation_dir, 'label'+str(i)+'.hdf5'), layer)
        layer_activations_i = torch.tensor(layer_activations_i)
        layer_activations_i = F.relu(layer_activations_i[:, :, :, :]).numpy()
        num_patches = layer_activations_i.shape[2]
        layer_activations_i_new = []
        for j in range(num_patches):
            for k in range(num_patches):
                layer_activations_i_new.append(layer_activations_i[:,:,j,k])
        layer_activations_i = np.vstack([l for l in layer_activations_i_new])
        label_i = np.repeat(i, len(layer_activations_i)).reshape(-1,1)
        layer_activations_i = np.hstack([label_i, layer_activations_i])
        layer_activations.append(layer_activations_i)    

    layer_activations = np.vstack([layer_activations_i for layer_activations_i in layer_activations])
    layer_activations_df = pd.DataFrame(layer_activations)
    
    cols = np.array(['label'])
    cols = np.concatenate((cols,np.arange(1,layer_activations.shape[1]).astype("str")))
    
    layer_activations_df.columns = cols 
    layer_activations_df['label'] = [names[int(layer_activations_df['label'].iloc[i])] for i in range(len(layer_activations_df))]
    print(layer_activations_df.shape)

    if layer_activations_df.shape[0] > 800000:
        selected_indices = np.random.choice(layer_activations_df.shape[0], 800000)
        layer_activations_df = layer_activations_df.iloc[selected_indices, :]

    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[4])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, 1:])
    print("eps", eps)
            
    min_samples = 5
    
    get_mapper_graph(
        df=layer_activations_df, 
        interval=interval, 
        overlap=overlap, 
        eps=eps, 
        min_samples=min_samples, 
        output_dir=str(graph_dir), 
        output_fname='full_batches_'+layer, 
        is_parallel=False
    )
