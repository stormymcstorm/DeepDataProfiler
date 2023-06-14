#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 20:47:33 2021

"""

import os
import sys
import collections
from functools import partial
from glob import glob
import argparse
import pathlib

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from pynndescent import NNDescent
from matplotlib import pyplot as plt

from get_knn import elbow_eps
from mapper_interactive.mapper_CLI import get_mapper_graph

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
        '--dataset-dir', type=pathlib.Path, default='datasets',
        help='The directory to read/download datasets from/to.'
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

    layer = args.layer
    interval = args.interval
    overlap = args.overlap

    df_dir = str(args.dataset_dir / "cifar10_single_batch_df")
    
    print("collection single activations for", layer)
    
    layer_activations_df = pd.read_csv(df_dir+"/train_single_batch_"+layer+".csv")
    print(layer_activations_df.shape)

    try:
      eps = float(sys.argv[4])
    except:
      eps = elbow_eps(layer_activations_df.iloc[:, 1:])
    print("eps", eps)
            
    min_samples = 5
    # interval = 40
    # overlap = 30
    
    output_dir = args.graph_dir / 'mapper_graphs' / 'single_batches'
    output_fname = 'single_batch_'+layer
    
    get_mapper_graph(layer_activations_df, interval, overlap, eps, min_samples, output_dir, output_fname)
    
    

