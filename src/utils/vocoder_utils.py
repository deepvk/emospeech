import glob
import os

import numpy as np
import torch
from loguru import logger
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath):
    logger.info(f"Checking {filepath} is file...")
    assert os.path.isfile(filepath)
    logger.info(f"Loading {filepath}...")
    checkpoint_dict = torch.load(filepath)
    logger.info("Complete.")
    return checkpoint_dict


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def synthesize_wav_from_mel(mel: torch.Tensor, vocoder, stft) -> np.ndarray:
    mel = mel.transpose(0, 1)
    vocoder.to(mel.device)
    spec, phase = vocoder(mel.unsqueeze(0))
    wav = stft.inverse(spec, phase).squeeze(0).squeeze(0).detach().cpu().numpy()

    return wav
