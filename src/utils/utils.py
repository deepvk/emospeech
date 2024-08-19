import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger
from scipy.io import wavfile


def compute_overall_mos(d: dict) -> tuple[float, float]:
    return np.mean(list(d.values())), np.std(list(d.values()))


def compute_mos_per_speaker(d: dict) -> dict:
    res = {}
    res = defaultdict(lambda: [], res)
    for basename, score in zip(list(d.keys()), list(d.values())):
        speaker, _, _ = basename.split("_")
        res[speaker].append(score)
    mos_dict = {}
    for k, v in zip(list(res.keys()), list(res.values())):
        mos_dict[k] = np.mean(v)
    return mos_dict


def write_txt(txt_path: Path, data: list) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        for m in data:
            f.write(m + "\n")


def set_up_logger(filename: str) -> None:
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(filename, format=fmt)


def crash_with_msg(message: str) -> None:
    logger.error(message)
    sys.exit(1)


def write_wav(path: Union[Path, str], wav: np.ndarray, sample_rate=16000) -> None:
    wavfile.write(path, sample_rate, wav)


def write_json(d: dict, path: Union[Path, str]) -> None:
    with open(path, "a") as f:
        json.dump(d, f)
        f.write("\n")
