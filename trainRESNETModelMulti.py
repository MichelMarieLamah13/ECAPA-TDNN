"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import argparse, glob, os, torch, warnings, time
import sys

import numpy as np
import yaml
from torch.utils.data import DataLoader

from RESNETModel import RESNETModelMulti
from tools import *
from dataLoader import train_loader, TrainDatasetMulti
from ECAPAModel import ECAPAModel, ECAPAModelMulti


def init_eer(score_path):
    errs = {}
    with open(score_path) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            parteer = line.split(',')[-1]
            parteer = parteer.split(' ')[-1]
            parteer = parteer.replace('%', '')
            parteer = float(parteer)
            if 'File' not in line:
                if 'mean' not in errs:
                    errs['mean'] = [parteer]
                else:
                    errs['mean'].append(parteer)
            else:
                fname = line.split(',')[0]
                fname = fname.split()[-1].strip()
                if fname not in errs:
                    errs[fname] = [parteer]
                else:
                    errs[fname].append(parteer)
    return errs


def read_config(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
        for key, item in config.items():
            setattr(args, key, item['value'])

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RESNET_trainer Multi corpus")
    parser.add_argument('--config',
                        type=str,
                        default="config_resnet_multi.yml",
                        help='Configuration file')

    # Initialization
    warnings.simplefilter("ignore")
    # torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    args = read_config(args)
    args = init_args(args)

    # Define the data loader
    trainloader = TrainDatasetMulti(**vars(args))
    trainLoader = DataLoader(trainloader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu,
                             drop_last=True, pin_memory=True)

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Get all eval_list
    eval_list = args.eval_list.strip().split('\n')
    eval_list = [el.strip() for el in eval_list if len(el.strip()) > 0]
    eval_path = args.eval_path.strip().split('\n')
    eval_path = [ep.strip() for ep in eval_path if len(ep.strip()) > 0]
    # Only do evaluation, the initial_model is necessary
    result = []
    if args.eval:
        s = RESNETModelMulti(**vars(args))
        print(f"Model {args.initial_model} loaded from previous state!", flush=True)
        s.load_parameters(args.initial_model)
        for i, eval_list_ in enumerate(eval_list):
            eval_path_ = eval_path[i]
            fname = eval_list_.split('/')[-1]
            EER, minDCF = s.eval_network(eval_list=eval_list_, eval_path=eval_path_, n_cpu=args.n_cpu)
            result.append((fname, EER, minDCF))

        for fname, eer, mindcf in result:
            print(f"File {fname}, EER {eer:2.2f}%, minDCF {mindcf:.4f}%", flush=True)

        eers = [eer for _, eer, _ in result]
        mindcfs = [mindcf for _, _, mindcf in result]
        print(f"Mean: EER  {np.mean(eers):2.2f}%, minDCF {np.mean(mindcfs):.4f}%", flush=True)
        quit()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print(f"Model {args.initial_model} loaded from previous state!", flush=True)
        s = RESNETModelMulti(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
        EERs = {}

    # Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print(f"Model {modelfiles[-1]} loaded from previous state!", flush=True)
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = RESNETModelMulti(**vars(args))
        s.load_parameters(modelfiles[-1])
        EERs = init_eer(args.score_save_path)

    # Otherwise, system will train from scratch
    else:
        epoch = 1
        s = RESNETModelMulti(**vars(args))
        EERs = {}

    score_file = open(args.score_save_path, "a+")

    while True:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)
        result = []
        # Evaluation every [test_step] epochs
        if epoch % args.test_step == 0:
            s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch, delete=True)
            sum_eer = 0
            sum_mindcf = 0
            for i, eval_list_ in enumerate(eval_list):
                eval_list_ = eval_list_.strip()
                eval_path_ = eval_path[i].strip()
                fname = eval_list_.split('/')[-1].strip()
                eer, mindcf = s.eval_network(eval_list=eval_list_, eval_path=eval_path_, n_cpu=args.n_cpu)
                EERs[fname].append(eer)
                result.append((fname, eer, mindcf, min(EERs[fname])))
                sum_mindcf += mindcf
                sum_eer += eer
                score_file.write(
                    f"File {fname}, {epoch} epoch, LR {lr}, LOSS {loss}, ACC {acc:2.2f}%, EER {eer:2.2f}%, "
                    f"bestEER {min(EERs[fname]):2.2f}%\n")
                score_file.flush()

            for fname, eer, mindcf, besteer in result:
                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} File {fname}, {epoch} epoch, ACC {acc:2.2f}%, "
                    f"EER {eer:2.2f}%, bestEER {besteer:2.2f}%",
                    flush=True)
            mean_eer = sum_eer / len(eval_path)
            mean_dcf = sum_mindcf / len(eval_path)
            EERs['mean'].append(mean_eer)
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}  Mean, {epoch} epoch, ACC {acc:2.2f}%, "
                f"EER {mean_eer:2.2f}%, bestEER {min(EERs['mean']):2.2f}%",
                flush=True)
            score_file.write(
                f"{epoch} epoch, LR {lr}, LOSS {loss}, ACC {acc:2.2f}%, EER {mean_eer:2.2f}%, "
                f"bestEER {min(EERs['mean']):2.2f}%\n")
            score_file.flush()
            if mean_eer <= min(EERs['mean']):
                s.save_parameters(args.model_save_path + "/best.model")

        if epoch >= args.max_epoch:
            quit()

        epoch += 1
