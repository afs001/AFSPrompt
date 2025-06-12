# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: Utilities for layer definitions
# ------------------------------------------------------------------------------ #
from PIL import Image, ImageOps
from torch import nn
from math import ceil

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


def flatten(x):
    x = x.view(x.shape[0], x.shape[1], -1)\
        .permute(0, 2, 1).contiguous()
    return x


def unflatten(x, shape):
    x = x.permute(0, 2, 1).contiguous()\
        .view(x.shape[0], -1, shape[0], shape[1])
    return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def Pad():
    def _pad(image):
        W, H = image.size # debugged
        if H < W:
            pad_H = ceil((W - H) / 2)
            pad_W = 0
        else:
            pad_H = 0
            pad_W = ceil((H - W) / 2)
        img = ImageOps.expand(image, border=(pad_W, pad_H, pad_W, pad_H), fill=0)
        # print(img.size)
        return img
    return _pad

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def identity(x):
    return x

def _transform(n_px, pad=False, crop=False):
    return Compose([
        Pad() if pad else identity,
        Resize([n_px, n_px], interpolation=BICUBIC),
        CenterCrop(n_px) if crop else identity,
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])