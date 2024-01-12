import pdb
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x, learnable_weights):
        pdb.set_trace()
        with torch.no_grad():
            x = self.processor(x, return_tensor='pt', sampling_rate=16_000)
            x = x.input_values[0]
            x = torch.tensor(x)
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        result = torch.zeros_like(hidden_states[0])
        for i, hidden in enumerate(hidden_states):
            weights = learnable_weights[i]
            result += weights * hidden

        return result
