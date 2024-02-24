"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import argparse, glob, os, torch, warnings, time
import sys

import yaml
from torch.utils.data import DataLoader

from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel


def init_eer(score_path):
    errs = []
    with open(score_path) as file:
        lines = file.readlines()
        for line in lines:
            parteer = line.split(',')[-1]
            parteer = parteer.split(' ')[-1]
            parteer = parteer.replace('%', '')
            parteer = float(parteer)
            if parteer not in errs:
                errs.append(parteer)
    return errs


def read_config(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
        for key, item in config.items():
            setattr(args, key, item['value'])

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECAPA_trainer Multi corpus")
    parser.add_argument('--config',
                        type=str,
                        default="config.yml",
                        help='Configuration file')

    args = parser.parse_args()
    args = read_config(args)
    args = init_args(args)
