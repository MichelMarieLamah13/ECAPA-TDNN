'''
This part is used to train the speaker model and evaluate the performances
'''
import glob

import numpy as np
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
import random
from torch.nn.parallel import DistributedDataParallel as DDP

from wav2vec2 import CustomWav2Vec2Model
from wavlm import CustomWavLMModel


def collate_fn(batch):
    # Separate filenames, data_1, and data_2
    filenames, data_1, original_lengths_1, data_2 = zip(*batch)
    max_length = np.max(original_lengths_1)
    data_1_padded = [torch.nn.functional.pad(seq, (0, max_length - seq.size(1))) for seq in data_1]
    return filenames, data_1_padded, original_lengths_1, data_2


class EmbeddingsDataset(Dataset):
    def __init__(self, files, eval_path):
        self.files = files
        self.eval_path = eval_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        audio, _ = soundfile.read(os.path.join(self.eval_path, file))
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0))

        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = numpy.stack(feats, axis=0).astype(numpy.float64)
        data_2 = torch.FloatTensor(feats)

        return file, data_1, data_1.shape[1], data_2


class ScoresDataset(Dataset):
    def __init__(self, lines, embeddings):
        self.lines = lines
        self.embeddings = embeddings

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        part0, part1, part2 = line.split()
        embedding_11, embedding_12 = self.embeddings[part1]
        embedding_21, embedding_22 = self.embeddings[part2]

        return embedding_11, embedding_12, embedding_21, embedding_22, int(part0)


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, feat_type, feat_dim, is_2d, model_name, **kwargs):
        super(ECAPAModel, self).__init__()

        self.learnable_weights = None
        self.is_2d = is_2d
        if feat_type == 'wav2vec2':
            wav2vec2 = CustomWav2Vec2Model(model_name=model_name)
            n_layers, feat_dim = wav2vec2.get_output_dim()
            if self.is_2d:
                self.learnable_weights = nn.Parameter(
                    torch.zeros(n_layers, feat_dim))  # 13 couches: CNN + 12 transformers
            else:
                self.learnable_weights = nn.Parameter(torch.ones(n_layers))
        elif feat_type == 'wavlm':
            wavlm = CustomWavLMModel(model_name=model_name)
            n_layers, feat_dim = wavlm.get_output_dim()
            self.learnable_weights = nn.Parameter(torch.ones(n_layers))

        # ECAPA-TDNN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim, model_name=model_name).to(
            self.device)
        # self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim)
        # Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).to(self.device)
        # self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
            self.zero_grad()
            labels = torch.LongTensor(labels).to(self.device)
            # labels = torch.LongTensor(labels)
            if self.learnable_weights is not None:
                speaker_embedding = self.speaker_encoder(data.to(self.device), aug=True,
                                                         learnable_weights=self.learnable_weights,
                                                         is_2d=self.is_2d)
            else:
                speaker_embedding = self.speaker_encoder(data.to(self.device), aug=True)

            # speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            print(time.strftime("%m-%d %H:%M:%S") + \
                  " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                  " Loss: %.5f, ACC: %2.2f%%" % (loss / num, top1 / index * len(labels)))
            sys.stdout.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path, n_cpu=5):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        print("BEGIN split", flush=True)
        for line in tqdm.tqdm(lines):
            _, part1, part2 = line.split()
            files.append(part1)
            files.append(part2)
        setfiles = list(set(files))
        setfiles.sort()
        print("END split", flush=True)

        print("BEGIN embeddings", flush=True)
        emb_dataset = EmbeddingsDataset(setfiles, eval_path)
        emb_loader = DataLoader(emb_dataset, batch_size=100, num_workers=n_cpu, collate_fn=collate_fn)
        for idx, batch in tqdm.tqdm(enumerate(emb_loader, start=1), total=len(emb_loader)):
            all_file, all_data_1, all_lengths_1, all_data_2 = batch
            for i in range(len(all_file)):
                file = all_file[i]
                length_1 = all_lengths_1[i]
                data_1 = all_data_1[i][:, :length_1]
                data_1 = data_1.to(self.device)
                data_2 = all_data_2[i].to(self.device)
                with torch.no_grad():
                    if self.learnable_weights is None:
                        embedding_1 = self.speaker_encoder(data_1, aug=False)
                        embedding_2 = self.speaker_encoder(data_2, aug=False)
                    else:
                        embedding_1 = self.speaker_encoder(data_1, aug=False,
                                                           learnable_weights=self.learnable_weights,
                                                           is_2d=self.is_2d)
                        embedding_2 = self.speaker_encoder(data_2, aug=False,
                                                           learnable_weights=self.learnable_weights,
                                                           is_2d=self.is_2d)
                    embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        print("END embeddings", flush=True)

        print("BEGIN scores", flush=True)
        for line in tqdm.tqdm(lines):
            part0, part1, part2 = line.split()
            embedding_11, embedding_12 = embeddings[part1]
            embedding_21, embedding_22 = embeddings[part2]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(part0))

        print("END scores", flush=True)

        print("BEGIN final score", flush=True)
        # Coumpute EER and minDCF
        EER, minDCF = 0, 0
        if len(scores) > 0 and len(labels) > 0:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            print("Pas de ligne correcte", flush=True)

        print("END final score", flush=True)

        return EER, minDCF

    def save_parameters(self, path, delete=False):
        if delete:
            folder = os.path.dirname(path)
            old_files = glob.glob(f'{folder}/model_0*.model')
            for file in old_files:
                os.remove(file)
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=self.device)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


class ECAPAModelDDP(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, feat_type, feat_dim, is_2d, model_name, gpu_id,
                 **kwargs):
        super(ECAPAModelDDP, self).__init__()

        # CUDA part
        self.gpu_id = gpu_id
        torch.cuda.set_device(self.gpu_id)
        torch.cuda.empty_cache()
        self.disable_tqdm = self.gpu_id != 0

        self.learnable_weights = None
        self.feat_type = feat_type
        self.is_2d = is_2d
        self.find_unused_parameters = False
        if self.feat_type == 'wav2vec2':
            self.find_unused_parameters = True
            wav2vec2 = CustomWav2Vec2Model(model_name=model_name)
            n_layers, feat_dim = wav2vec2.get_output_dim()
            if self.is_2d:
                self.learnable_weights = nn.Parameter(
                    torch.zeros(n_layers, feat_dim))  # 13 couches: CNN + 12 transformers
            else:
                self.learnable_weights = nn.Parameter(torch.ones(n_layers))
        elif self.feat_type == 'wavlm':
            wavlm = CustomWavLMModel(model_name=model_name)
            n_layers, feat_dim = wavlm.get_output_dim()
            self.learnable_weights = nn.Parameter(torch.ones(n_layers))

        # ECAPA-TDNN
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim, model_name=model_name).to(
            self.gpu_id)
        self.speaker_encoder = DDP(
            self.speaker_encoder,
            device_ids=[self.gpu_id],
            find_unused_parameters=self.find_unused_parameters
        )
        # self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim)
        # Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).to(self.gpu_id)
        # self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        self.print_info(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def print_info(self, content):
        if self.gpu_id == 0:
            print(content, flush=True)

    def train_network(self, epoch, loader, sampler):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        sampler.set_epoch(epoch)
        for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1), total=len(loader), disable=self.disable_tqdm):
            self.zero_grad()
            labels = torch.LongTensor(labels).to(self.gpu_id)
            # labels = torch.LongTensor(labels)
            speaker_embedding = self.speaker_encoder(data.to(self.gpu_id), aug=True,
                                                     learnable_weights=self.learnable_weights,
                                                     is_2d=self.is_2d)

            # speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            self.print_info(time.strftime("%m-%d %H:%M:%S") + \
                            " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                            " Loss: %.5f, ACC: %2.2f%%" % (loss / num, top1 / index * len(labels)))
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path, n_cpu=5):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        self.print_info("BEGIN filter")
        filtered_lines = []
        for line in tqdm.tqdm(lines, disable=self.disable_tqdm):
            _, part1, part2 = line.split()
            path1 = os.path.join(eval_path, part1)
            path2 = os.path.join(eval_path, part2)
            if os.path.exists(path1) and os.path.exists(path2):
                filtered_lines.append(line)

        lines = filtered_lines
        self.print_info("END filter")

        self.print_info("BEGIN split")
        for line in tqdm.tqdm(lines, disable=self.disable_tqdm):
            _, part1, part2 = line.split()
            files.append(part1)
            files.append(part2)
        setfiles = list(set(files))
        setfiles.sort()
        self.print_info("END split")

        self.print_info("BEGIN embeddings")

        emb_dataset = EmbeddingsDataset(setfiles, eval_path)
        emb_loader = DataLoader(emb_dataset, batch_size=100, num_workers=n_cpu, collate_fn=collate_fn, shuffle=False)
        for idx, batch in tqdm.tqdm(enumerate(emb_loader, start=1), total=len(emb_loader), disable=self.disable_tqdm):
            all_file, all_data_1, all_lengths_1, all_data_2 = batch
            for i in range(len(all_file)):
                file = all_file[i]
                length_1 = all_lengths_1[i]
                data_1 = all_data_1[i][:, :length_1]
                data_1 = data_1.to(self.gpu_id)
                data_2 = all_data_2[i].to(self.gpu_id)
                with torch.no_grad():
                    embedding_1 = self.speaker_encoder(data_1, aug=False,
                                                       learnable_weights=self.learnable_weights,
                                                       is_2d=self.is_2d)
                    embedding_2 = self.speaker_encoder(data_2, aug=False,
                                                       learnable_weights=self.learnable_weights,
                                                       is_2d=self.is_2d)

                    embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        self.print_info("END embeddings")

        self.print_info("BEGIN scores")
        for line in tqdm.tqdm(lines, disable=self.disable_tqdm):
            part0, part1, part2 = line.split()
            embedding_11, embedding_12 = embeddings[part1]
            embedding_21, embedding_22 = embeddings[part2]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(part0))

        self.print_info("END scores")

        self.print_info("BEGIN final score")
        # Coumpute EER and minDCF
        EER, minDCF = 0, 0
        if len(scores) > 0 and len(labels) > 0:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            self.print_info(f"Pas de ligne correcte")

        self.print_info("END final score")

        return EER, minDCF

    def save_parameters(self, path, delete=False):
        if delete:
            folder = os.path.dirname(path)
            old_files = glob.glob(f'{folder}/model_0*.model')
            for file in old_files:
                os.remove(file)
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location="cpu")
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace('speaker_encoder.', 'speaker_encoder.module.')
                if name not in self_state:
                    self.print_info("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                self.print_info("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


class ECAPAModelMulti(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, feat_type, feat_dim, is_2d, model_name, **kwargs):
        super(ECAPAModelMulti, self).__init__()

        self.learnable_weights = None
        self.is_2d = is_2d
        if feat_type == 'wav2vec2':
            wav2vec2 = CustomWav2Vec2Model(model_name=model_name)
            n_layers, feat_dim = wav2vec2.get_output_dim()
            if self.is_2d:
                self.learnable_weights = nn.Parameter(
                    torch.zeros(n_layers, feat_dim))  # 13 couches: CNN + 12 transformers
            else:
                self.learnable_weights = nn.Parameter(torch.ones(n_layers))
        elif feat_type == 'wavlm':
            wavlm = CustomWavLMModel(model_name=model_name)
            n_layers, feat_dim = wavlm.get_output_dim()
            self.learnable_weights = nn.Parameter(torch.ones(n_layers))

        # ECAPA-TDNN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim, model_name=model_name).to(
            self.device)
        # self.speaker_encoder = ECAPA_TDNN(C=C, feat_type=feat_type, feat_dim=feat_dim)
        # Classifier
        n_class = n_class.strip().split('\n')
        n_class = [nc.strip() for nc in n_class if len(nc.strip()) > 0]
        self.speaker_loss = {}
        for i, n_class_ in enumerate(n_class):
            n_class_ = int(n_class_.strip())
            self.speaker_loss[i] = AAMsoftmax(n_class=n_class_, m=m, s=s).to(self.device)
        # self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        total_nloss = torch.tensor(0)
        for num, batch in tqdm.tqdm(enumerate(loader, start=1), total=len(loader)):
            self.zero_grad()
            total_loss = torch.tensor(0.0).to(self.device)  # Initialize total loss tensor
            total_prec = 0  # Initialize total precision
            total_index = 0  # Initialize total index
            for idx_loss, speaker_loss_ in enumerate(self.speaker_loss.values()):
                data = batch[idx_loss * 2]
                labels = torch.LongTensor(batch[idx_loss * 2 + 1]).to(self.device)
                if self.learnable_weights is not None:
                    speaker_embedding = self.speaker_encoder(data.to(self.device), aug=True,
                                                             learnable_weights=self.learnable_weights,
                                                             is_2d=self.is_2d)
                else:
                    speaker_embedding = self.speaker_encoder(data.to(self.device), aug=True)

                nloss, prec = speaker_loss_(speaker_embedding, labels)
                total_loss += nloss
                total_index += len(labels)
                total_prec += prec

            total_nloss.backward()
            self.optim.step()

            loss += total_loss.item()
            top1 += total_prec
            index += total_index

            print(time.strftime("%m-%d %H:%M:%S") + \
                  " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                  " Loss: %.5f, ACC: %2.2f%%" % (loss / num, top1 / index * len(labels)), flush=True)
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path, n_cpu=5):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        print("BEGIN split", flush=True)
        for line in tqdm.tqdm(lines):
            _, part1, part2 = line.split()
            files.append(part1)
            files.append(part2)
        setfiles = list(set(files))
        setfiles.sort()
        print("END split", flush=True)

        print("BEGIN embeddings", flush=True)

        emb_dataset = EmbeddingsDataset(setfiles, eval_path)
        emb_loader = DataLoader(emb_dataset, batch_size=100, num_workers=n_cpu, collate_fn=collate_fn)
        for idx, batch in tqdm.tqdm(enumerate(emb_loader, start=1), total=len(emb_loader)):
            all_file, all_data_1, all_lengths_1, all_data_2 = batch
            for i in range(len(all_file)):
                file = all_file[i]
                length_1 = all_lengths_1[i]
                data_1 = all_data_1[i][:, :length_1]
                data_1 = data_1.to(self.device)
                data_2 = all_data_2[i].to(self.device)
                with torch.no_grad():
                    if self.learnable_weights is None:
                        embedding_1 = self.speaker_encoder(data_1, aug=False)
                        embedding_2 = self.speaker_encoder(data_2, aug=False)
                    else:
                        embedding_1 = self.speaker_encoder(data_1, aug=False,
                                                           learnable_weights=self.learnable_weights,
                                                           is_2d=self.is_2d)
                        embedding_2 = self.speaker_encoder(data_2, aug=False,
                                                           learnable_weights=self.learnable_weights,
                                                           is_2d=self.is_2d)
                    embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        print("END embeddings", flush=True)

        print("BEGIN scores", flush=True)
        sys.stdout.flush()
        for line in tqdm.tqdm(lines):
            part0, part1, part2 = line.split()
            embedding_11, embedding_12 = embeddings[part1]
            embedding_21, embedding_22 = embeddings[part2]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(part0))

        print("END scores", flush=True)

        print("BEGIN final score", flush=True)
        # Coumpute EER and minDCF
        EER, minDCF = 0, 0
        if len(scores) > 0 and len(labels) > 0:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            print(f"Pas de ligne correcte", flush=True)

        print("END final score", flush=True)

        return EER, minDCF

    def save_parameters(self, path, delete=False):
        if delete:
            folder = os.path.dirname(path)
            old_files = glob.glob(f'{folder}/model_0*.model')
            for file in old_files:
                os.remove(file)
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=self.device)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
