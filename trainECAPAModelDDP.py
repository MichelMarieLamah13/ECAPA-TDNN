"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import argparse, glob, os, torch, warnings, time
import sys

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler

from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel, ECAPAModelDDP
import torch.multiprocessing as mp

torch.multiprocessing.set_sharing_strategy('file_system')


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


def ddp_setup(rank: int, world_size: int, master_port: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port  # select any idle port on your machine

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def print_info(rank, content):
    if rank == 0:
        print(content, flush=True)


def main_ddp(
        rank: int,
        world_size: int
):
    base_path = "./../kiwano-project/recipes/resnet"
    parser = argparse.ArgumentParser(description="ECAPA_trainer")
    # Training Settings
    parser.add_argument('--num_frames', type=int, default=200,
                        help='Duration of the input segments, eg: 200 for 2 second')
    parser.add_argument('--max_epoch', type=int, default=80, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=400, help='Batch size')
    parser.add_argument('--n_cpu', type=int, default=20, help='Number of loader threads')
    parser.add_argument('--test_step', type=int, default=1, help='Test and save every [test_step] epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument("--lr_decay", type=float, default=0.97, help='Learning rate decay every [test_step] epochs')

    # Training and evaluation path/lists, save path
    parser.add_argument('--train_list', type=str, default=f"{base_path}/db/voxceleb2/train_list.txt",
                        help='The path of the training list, '
                             'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
    parser.add_argument('--train_path', type=str, default=f"{base_path}/db/voxceleb2/wav",
                        help='The path of the training data, eg:"data/voxceleb2" in my case')
    parser.add_argument('--eval_list', type=str, default=f"{base_path}/db/voxceleb1/veri_test2.txt",
                        help='The path of the evaluation list: veri_test2.txt, list_test_all2.txt, list_test_hard2.txt'
                             'veri_test2.txt comes from '
                             'https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('--eval_path', type=str, default=f"{base_path}/db/voxceleb1/wav",
                        help='The path of the evaluation data, eg:"data/voxceleb1/" in my case')
    parser.add_argument('--musan_path', type=str, default=f"{base_path}/db/musan_split",
                        help='The path to the MUSAN set, eg:"data/musan_split" in my case')
    parser.add_argument('--rir_path', type=str, default=f"{base_path}/db/rirs_noises/RIRS_NOISES/simulated_rirs",
                        help='The path to the RIR set, eg:"data/RIRS_NOISES/simulated_rirs" in my case')
    parser.add_argument('--save_path', type=str, default="exps/exp1", help='Path to save the score.txt and models')
    parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')

    # Model and Loss settings
    parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
    parser.add_argument('--m', type=float, default=0.2, help='Loss margin in AAM softmax')
    parser.add_argument('--s', type=float, default=30, help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int, default=5994, help='Number of speakers')
    parser.add_argument('--feat_type', type=str, default='fbank', help='Type of features: fbank, wav2vec2')
    parser.add_argument('--feat_dim', type=int, default=80, help='Dim of features: fbank(80), wav2vec2(768)')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h',
                        help='facebook/wav2vec2-base-960h, facebook/wav2vec2-large-960h'
                             'facebook/wav2vec2-large-robust-ft-libri-960h, facebook/wav2vec2-large-960h-lv60-self')
    parser.add_argument('--is_2d', dest='is_2d', action='store_true', help='2d learneable weight')
    parser.add_argument('--master_port', type=str, default="54323", help='Master port')

    # Command
    parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

    # Initialization
    warnings.simplefilter("ignore")
    # torch.multiprocessing.set_start_method('spawn', force=True)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    args = parser.parse_args()
    args = init_args(args)
    args.gpu_id = rank
    # Define the data loader
    ddp_setup(rank, world_size, args.master_port)

    train_data = train_loader(**vars(args))
    training_sampler = DistributedSampler(train_data)
    trainLoader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=training_sampler,
        num_workers=args.n_cpu,
        drop_last=True)

    # Search for the exist models
    modelfiles = glob.glob('%s/model_0*.model' % args.model_save_path)
    modelfiles.sort()

    # Only do evaluation, the initial_model is necessary
    if args.eval:
        s = ECAPAModelDDP(**vars(args))
        print("Model %s loaded from previous state!" % args.initial_model)
        sys.stdout.flush()
        s.load_parameters(args.initial_model)
        EER, minDCF = s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)
        print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        sys.stdout.flush()
        quit()

    # If initial_model is exist, system will train from the initial_model
    if args.initial_model != "":
        print("Model %s loaded from previous state!" % args.initial_model)
        sys.stdout.flush()
        s = ECAPAModelDDP(**vars(args))
        s.load_parameters(args.initial_model)
        epoch = 1
        EERs = []

    # Otherwise, system will try to start from the saved model&epoch
    elif len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        sys.stdout.flush()
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ECAPAModelDDP(**vars(args))
        s.load_parameters(modelfiles[-1])
        EERs = init_eer(args.score_save_path)

    # Otherwise, system will train from scratch
    else:
        epoch = 1
        s = ECAPAModelDDP(**vars(args))
        EERs = []

    if rank == 0:
        score_file = open(args.score_save_path, "a+")

    while True:
        # Training for one epoch
        loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader, sampler=training_sampler)

        # Evaluation every [test_step] epochs
        if rank == 0:
            if epoch % args.test_step == 0:
                s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch, delete=True)
                EERs.append(s.eval_network(eval_list=args.eval_list, eval_path=args.eval_path, n_cpu=args.n_cpu)[0])
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%" % (epoch, acc, EERs[-1], min(EERs)))
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" % (
                    epoch, lr, loss, acc, EERs[-1], min(EERs)))
                score_file.flush()
                if EERs[-1] <= min(EERs):
                    s.save_parameters(args.model_save_path + "/best.model")

            if epoch >= args.max_epoch:
                destroy_process_group()  # clean up
                quit()

        epoch += 1


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_ddp,
        args=(world_size,),
        nprocs=world_size,  # Total number of process = number of gpus
    )
