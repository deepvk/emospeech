import torch
import numpy as np
import torch.nn.functional as F

from typing import Optional, List


def pad_or_trim_mel(
    mel: torch.Tensor, target_len: int, pad_idx: int = 0
) -> torch.Tensor:
    mel = mel.detach().cpu().transpose(0, 1)
    if mel.shape[1] >= target_len:
        return mel[:, :target_len]
    else:
        return F.pad(
            mel, (0, target_len - np.shape(mel)[0]), mode="constant", value=pad_idx
        )


def get_mask_from_lengths(
    lengths: torch.Tensor, device: torch.device
) -> torch.BoolTensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    lengths = lengths.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values: Optional[np.ndarray], durations: Optional[np.ndarray]) -> np.ndarray:
    out = []
    for value, d in zip(values, durations):
        out.extend([value] * max(0, int(d)))

    return np.array(out)


def pad_1d(inputs: Optional[List[np.ndarray]], pad_value=0) -> np.ndarray:
    def _pad(x: np.ndarray, max_length: int, pad_value: int):
        x_padded = np.pad(
            x, (0, max_length - x.shape[0]), mode="constant", constant_values=pad_value
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([_pad(x, max_len, pad_value) for x in inputs])

    return padded


def pad_2d(inputs: List[np.ndarray], max_len=None, pad_value=0) -> np.ndarray:
    def _pad(x: np.ndarray, max_len: int) -> np.ndarray:
        if x.shape[0] > max_len:
            raise ValueError("not max_len")

        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=pad_value
        )
        return x_padded[:, : x.shape[1]]

    if not max_len:
        max_len = max(x.shape[0] for x in inputs)
    output = np.stack([_pad(x, max_len) for x in inputs])

    return output


def pad(input_tensor: List[torch.Tensor], mel_max_length=None) -> torch.Tensor:
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_tensor[i].size(0) for i in range(len(input_tensor))])

    out_list = []
    for i, batch in enumerate(input_tensor):
        if len(batch.shape) == 1:
            out_list.append(F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0))
        elif len(batch.shape) == 2:
            out_list.append(
                F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
            )
    out_padded = torch.stack(out_list)

    return out_padded
