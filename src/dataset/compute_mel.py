import numpy as np
import torch
import torch.nn as nn
import torchaudio

MIN_MEL_VALUE = 1e-05
PAD_MEL_VALUE = -11.52


class ComputeMelEnergy:
    def __init__(
        self,
        sample_rate,
        stft_length,
        hop_length,
        n_mel_channels,
        power=1,
        center=False,
        mel_fmin=0,
        mel_fmax=None,
        pad_value=PAD_MEL_VALUE,
        **_
    ):
        self.sample_rate = sample_rate
        self.stft_length = stft_length
        self.hop_length = hop_length
        self.n_mels = n_mel_channels
        self.f_min = mel_fmin
        self.f_max = mel_fmax
        self.hop_length = hop_length
        self.padding = stft_length - hop_length
        self.pad_value = pad_value

    def _get_mel_energy(self, wav: torch.Tensor):
        padded_wav = nn.functional.pad(
            wav, (self.padding // 2, self.padding // 2), mode="reflect"
        )
        spectrogram = torchaudio.functional.spectrogram(
            padded_wav,
            pad=0,
            power=1,
            center=False,
            onesided=True,
            normalized=False,
            n_fft=self.stft_length,
            win_length=self.stft_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.stft_length),
        )
        melscale = torchaudio.functional.melscale_fbanks(
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            n_freqs=self.stft_length // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        )
        mel_spectrogram = torch.matmul(
            spectrogram.transpose(-1, -2), melscale
        ).transpose(-1, -2)
        mel = self.log_mel(mel_spectrogram)
        energy = torch.norm(spectrogram, dim=1)
        return mel, energy

    @staticmethod
    def log_mel(mel):
        return torch.log(torch.clamp(mel, min=MIN_MEL_VALUE))

    def __call__(self, audio: np.ndarray):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        melspec, energy = self._get_mel_energy(audio)
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, energy
