'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN


class EmbeddingsDataset(Dataset):
    def __init__(self, files, eval_path, speaker_encoder):
        self.files = files
        self.eval_path = eval_path
        self.speaker_encoder = speaker_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        audio, _ = soundfile.read(os.path.join(self.eval_path, file))
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

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
        data_2 = torch.FloatTensor(feats).cuda()
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)

        return file, embedding_1, embedding_2


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
        # Compute the scores
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()

        return score, int(part0)


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        # ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        # Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

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
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        print("BEGIN filter")
        sys.stdout.flush()
        filtered_lines = []
        for line in lines:
            _, part1, part2 = line.split()
            path1 = os.path.join(eval_path, part1)
            path2 = os.path.join(eval_path, part2)
            if os.path.exists(path1) and os.path.exists(path2):
                filtered_lines.append(line)

        lines = filtered_lines
        print("END filter")
        sys.stdout.flush()

        print("BEGIN split")
        sys.stdout.flush()
        for line in tqdm.tqdm(lines):
            _, part1, part2 = line.split()
            files.append(part1)
            files.append(part2)
        setfiles = list(set(files))
        setfiles.sort()
        print("END split")
        sys.stdout.flush()

        print("BEGIN embeddings")
        sys.stdout.flush()
        # for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
        #     audio, _ = soundfile.read(os.path.join(eval_path, file))
        #     # Full utterance
        #     data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
        #
        #     # Spliited utterance matrix
        #     max_audio = 300 * 160 + 240
        #     if audio.shape[0] <= max_audio:
        #         shortage = max_audio - audio.shape[0]
        #         audio = numpy.pad(audio, (0, shortage), 'wrap')
        #     feats = []
        #     startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        #     for asf in startframe:
        #         feats.append(audio[int(asf):int(asf) + max_audio])
        #     feats = numpy.stack(feats, axis=0).astype(numpy.float64)
        #     data_2 = torch.FloatTensor(feats).cuda()
        #     # Speaker embeddings
        #     with torch.no_grad():
        #         embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
        #         embedding_1 = F.normalize(embedding_1, p=2, dim=1)
        #         embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
        #         embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        #     embeddings[file] = [embedding_1, embedding_2]
        emb_dataset = EmbeddingsDataset(setfiles, eval_path, self.speaker_encoder)
        emb_loader = DataLoader(emb_dataset, batch_size=100, num_workers=5)
        for idx, batch in tqdm.tqdm(enumerate(emb_loader, start=1), total=len(emb_loader)):
            for file, embedding_1, embedding_2 in batch:
                embeddings[file] = [embedding_1, embedding_2]
            print(f"Batch [{idx}/{len(emb_loader)}] DONE")

        scores, labels = [], []
        print("END embeddings")
        sys.stdout.flush()

        print("BEGIN scores")
        sys.stdout.flush()
        # for line in tqdm.tqdm(lines):
        #     part0, part1, part2 = line.split()
        #     embedding_11, embedding_12 = embeddings[part1]
        #     embedding_21, embedding_22 = embeddings[part2]
        #     # Compute the scores
        #     score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
        #     score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        #     score = (score_1 + score_2) / 2
        #     score = score.detach().cpu().numpy()
        #     scores.append(score)
        #     labels.append(int(part0))
        scores_dataset = ScoresDataset(lines, embeddings)
        scores_loader = DataLoader(scores_dataset, batch_size=50, num_workers=5)
        for idx, batch in tqdm.tqdm(enumerate(scores_loader, start=1), total=len(scores_loader)):
            for score, label in batch:
                scores.append(score)
                labels.append(label)
            print(f"Batch [{idx}/{len(emb_loader)}] DONE")

        print("END scores")
        sys.stdout.flush()

        print("BEGIN final score")
        sys.stdout.flush()
        # Coumpute EER and minDCF
        EER, minDCF = 0, 0
        if len(scores) > 0 and len(labels) > 0:
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        else:
            print(f"Pas de ligne correcte")
            sys.stdout.flush()

        print("END final score")
        sys.stdout.flush()

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
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
