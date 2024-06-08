import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

import torchaudio
from scipy.stats import skew, kurtosis
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

from model import PreEmphasis, FbankAug

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def pooling(x, mode='statistical'):
    """
        function that implement different kind of pooling
    """
    if mode == 'min':
        x, _ = x.min(dim=2)
    elif mode == 'max':
        x, _ = x.min(dim=2)
    elif mode == 'mean':
        x = x.mean(dim=2)
    elif mode == 'std':
        x = x.std(dim=2)
    elif mode == 'statistical':
        means = x.mean(dim=2)
        stds = x.std(dim=2)
        x = torch.cat([means, stds], dim=1)
    elif mode == 'std_kurtosis':
        stds = x.std(dim=2)
        kurtoses = kurtosis(x.detach().cpu(), axis=2, fisher=False)
        kurtoses = torch.from_numpy(kurtoses)
        kurtoses = kurtoses.to(stds.device)
        x = torch.cat([stds, kurtoses], dim=1)
    elif mode == 'std_skew':
        stds = x.std(dim=2)
        skews = skew(x.detach().cpu(), axis=2)
        skews = torch.from_numpy(skews)
        skews = skews.to(stds.device)
        x = torch.cat([stds, skews], dim=1)
    else:
        raise ValueError('Unexpected pooling mode.')

    return x


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        out = torch.cat(inputs, 1)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        return out

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        out = self.norm2(bottleneck_output)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        out = torch.cat(features, 1)
        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        emb_size (int) - embedding size
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
        stride (int) - stride for first convolution
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, emb_size=192, stride=2, pooling_mode='std',
                 features_per_frame=80, memory_efficient=False):

        super(DenseNet, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )
        self.specaug = FbankAug()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=stride,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.pooling_mode = pooling_mode
        pooling_size = 2 if self.pooling_mode in ['statistical', 'std_skew', 'std_kurtosis'] else 1
        before = 2 if stride == 2 else 0
        self.fc = nn.Linear(
            num_features * math.floor(features_per_frame * (0.5 ** ((len(block_config) - 1) + before))) * pooling_size,
            emb_size)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, aug):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = (x - torch.mean(x, dim=-1, keepdim=True))
            if aug:
                x = self.specaug(x)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.features(x)

        x = F.relu(x, inplace=True)

        x = x.transpose(2, 3)
        x = x.flatten(1, 2)
        x = pooling(x, self.pooling_mode)

        x = self.fc(x)
        return x


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, stride, pooling_mode):
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=num_init_features,
        stride=stride,
        pooling_mode=pooling_mode
    )
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet37(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-37 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride (int):
        pooling_mode (str):
    """
    return _densenet(
        arch='densenet37',
        growth_rate=32,
        block_config=(3, 4, 6, 3),
        num_init_features=64,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode)


def densenet121(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride (int):
        pooling_mode (str):
    """
    return _densenet(
        arch='densenet121',
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode)


def densenet161(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride(int):
        pooling_mode(str):
    """
    return _densenet(
        arch='densenet161',
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        num_init_features=96,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode
    )


def densenet169(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride(int):
        pooling_mode(str):
    """
    return _densenet(
        arch='densenet169',
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode
    )


def densenet201(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride(int):
        pooling_mode(str):
    """
    return _densenet(
        arch='densenet201',
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode
    )


def densenet264(pretrained=False, progress=True, stride=2, pooling_mode="std"):
    r"""Densenet-264 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        stride(int):
        pooling_mode(str):
    """
    return _densenet(
        arch='densenet264',
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        num_init_features=64,
        pretrained=pretrained,
        progress=progress,
        stride=stride,
        pooling_mode=pooling_mode
    )
