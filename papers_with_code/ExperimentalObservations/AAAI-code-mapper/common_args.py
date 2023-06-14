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
    
    def add_dataset_arg(self, choices=['CIFAR10', 'CIFAR100']):
        self.add_argument(
            '--dataset', choices=choices, type=str.upper, default=choices[0],
            help='The dataset used for training.'
        )

