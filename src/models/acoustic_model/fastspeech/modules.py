import json
import torch
import torch.nn as nn

from pathlib import Path
from typing import Tuple, Any

from src.utils.fastspeech_utils import pad, get_mask_from_lengths


class VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(config)
        self.length_regulator = LengthRegulator(config.device)
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)
        self.device = config.device
        self.n_egemap_features = config.n_egemap_features
        n_bins = config.variance_embedding_n_bins

        with open(str(Path(config.preprocessed_data_path) / "stats.json")) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]
            egemap_mins, egemap_maxs = stats["egemap"][:2]

        self.pitch_bins = nn.Parameter(
            torch.linspace(pitch_min, pitch_max, n_bins - 1), requires_grad=False
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(energy_min, energy_max, n_bins - 1), requires_grad=False
        )

        self.pitch_embedding = nn.Embedding(n_bins, config.transformer_encoder_hidden)
        self.energy_embedding = nn.Embedding(n_bins, config.transformer_encoder_hidden)

        if config.n_egemap_features:
            self.egemap_predictor = VariancePredictor(config)
            self.egemap_bins = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.linspace(egemap_mins[i], egemap_maxs[i], n_bins - 1),
                        requires_grad=False,
                    )
                    for i in range(config.n_egemap_features)
                ]
            )
            self.egemap_embeddings = nn.ModuleList(
                [
                    nn.Embedding(n_bins, config.transformer_encoder_hidden)
                    for i in range(config.n_egemap_features)
                ]
            )
        else:
            self.egemap_predictor = None
            self.egemap_bins = None
            self.egemap_embeddings = None

    def get_egemap_embedding(
        self, device, x, mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.egemap_predictor(x, mask)
        bs, seq_len = prediction.shape
        prediction = prediction[:, 0].unsqueeze(1).expand(bs, seq_len).contiguous()
        embedding = self.egemap_embeddings[0](
            torch.bucketize(prediction.to(device), self.egemap_bins[0].to(device))
        )
        for i in range(1, self.n_egemap_features):
            _prediction = prediction[:, i].unsqueeze(1).expand(bs, seq_len).contiguous()
            embedding += self.egemap_embeddings[i](
                torch.bucketize(_prediction.to(device), self.egemap_bins[i].to(device))
            )

        return prediction[:, : self.n_egemap_features], embedding

    def get_pitch_embedding(
        self, device, x, target, mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(
                torch.bucketize(target.to(device), self.pitch_bins.to(device))
            )
        else:
            embedding = self.pitch_embedding(
                torch.bucketize(prediction.to(device), self.pitch_bins.to(device))
            )
        return prediction, embedding

    def get_energy_embedding(
        self, device, x, target, mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(
                torch.bucketize(target.to(device), self.energy_bins.to(device))
            )
        else:
            embedding = self.energy_embedding(
                torch.bucketize(prediction.to(device), self.energy_bins.to(device))
            )
        return prediction, embedding

    def forward(
        self, device: str, x: torch.Tensor, src_mask: torch.Tensor, batch_dict: dict
    ) -> dict:
        max_mel_len = (
            torch.max(batch_dict["mel_lens"]).item()
            if batch_dict["mel_lens"] is not None
            else None
        )
        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            device, x, batch_dict["pitches"], src_mask
        )
        x = x + pitch_embedding
        energy_prediction, energy_embedding = self.get_energy_embedding(
            device, x, batch_dict["energies"], src_mask
        )
        x = x + energy_embedding
        if self.egemap_predictor:
            egemap_prediction, egemap_embedding = self.get_egemap_embedding(
                device, x, src_mask
            )
            x = x + egemap_embedding
        else:
            egemap_prediction = None
        if batch_dict["durations"] is not None:
            x, mel_len = self.length_regulator(x, batch_dict["durations"], max_mel_len)
            duration_rounded = batch_dict["durations"]
        else:
            duration_rounded = torch.clamp(
                torch.round(torch.exp(log_duration_prediction) - 1), min=0
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)

        var_adaptor_return_dict = {
            "output": x,
            "pitch_prediction": pitch_prediction,
            "energy_prediction": energy_prediction,
            "egemap_prediction": egemap_prediction,
            "log_duration_prediction": log_duration_prediction,
            "duration_rounded": duration_rounded,
            "mel_len": mel_len,
        }

        return var_adaptor_return_dict


class LengthRegulator(nn.Module):
    def __init__(self, device):
        super(LengthRegulator, self).__init__()
        self.device = device

    def _pad(self, x, duration, max_len):
        output, mel_len = [], []
        for batch, expand_target in zip(x, duration):  # b, t, c
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(self.device)

    @staticmethod
    def expand(batch, predicted):
        out = []
        for i, vec in enumerate(
            batch
        ):  # t_phoneme, c -> {t_i, c} /forall i t_mel, c -> t_mel, c; t_mel = /sum t_i
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self._pad(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    def __init__(self, config):
        super(VariancePredictor, self).__init__()

        self.conv_layer = nn.Sequential(
            Conv(
                config.transformer_encoder_hidden,
                config.variance_predictor_filter_size,
                kernel_size=config.variance_predictor_kernel_size,
                padding=(config.variance_predictor_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.LayerNorm(config.variance_predictor_filter_size),
            nn.Dropout(config.variance_predictor_dropout),
            Conv(
                config.variance_predictor_filter_size,
                config.variance_predictor_filter_size,
                kernel_size=config.variance_predictor_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.LayerNorm(config.variance_predictor_filter_size),
            nn.Dropout(config.variance_predictor_dropout),
        )

        self.linear_layer = nn.Linear(config.variance_predictor_filter_size, 1)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x
