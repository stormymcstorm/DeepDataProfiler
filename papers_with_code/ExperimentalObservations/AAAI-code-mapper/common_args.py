"""
A utility for defining common arguments to the scripts in this directory.
"""

import os
import argparse
import pathlib

def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)

class BaseParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        kwargs.setdefault('formatter_class', argparse.ArgumentDefaultsHelpFormatter)

        super().__init__(**kwargs)

        self.add_argument(
            '--cache-dir', type=pathlib.Path, default='out',
            help='The path to the directory where experimental results should be cached.'
        )
        

class TrainExtractParser(BaseParser):
    def __init__(
        self, 
        dataset_choices=['CIFAR10', 'CIFAR100'],
        workers=4,
        batch_size=128,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.add_argument(
            '--dataset', choices=dataset_choices, type=str.upper, default=dataset_choices[0],
            help='The dataset used for training.'
        )
        self.add_argument(
            '--workers', type=int, default=workers,
            help='How many subprocesses to use for data loading. 0 means the data will be loaded in the main process.'
        )
        self.add_argument(
            '--batch-size', type=int, default=batch_size,
            help='The batch size to load data with.'
        )

