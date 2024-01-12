import pdb
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor


def get_output_rep(hidden_states, learnable_weigths, n_layers, n_frames):
    sum_hiddens = torch.zeros(size=(1, n_frames))
    for layer in range(n_layers):
        sum_hiddens += learnable_weigths[layer] @ hidden_states[layer]
    return sum_hiddens.squeeze(dim=0)


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x):
        with torch.no_grad():
            x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
            x = x.input_values[0]
            x = torch.tensor(x)
            output = self.model(x)

        return output
