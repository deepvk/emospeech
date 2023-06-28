import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self, config):
        super(FastSpeech2Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.n_mels = config.n_mel_channels
        self.n_egemap_features = config.n_egemap_features

    def forward(
        self,
        device: torch.device,
        inputs: dict,
        predictions: dict,
        compute_mel_loss: bool = True,
    ) -> dict:
        phone_masks = ~predictions["phone_masks"].to(device)
        mel_masks = ~predictions["mel_masks"][:, :]
        mel_predictions = predictions["predicted_mel"]
        pitch_predictions = predictions["predicted_pitch"].masked_select(phone_masks)
        energy_predictions = predictions["predicted_energy"].masked_select(phone_masks)
        log_duration_predictions = predictions["predicted_log_durations"].masked_select(
            phone_masks
        )
        egemap_predictions = predictions["predicted_egemap"]
        mel_targets = inputs["mels"].detach()
        pitch_targets = inputs["pitches"].detach().masked_select(phone_masks)
        energy_targets = inputs["energies"].detach().masked_select(phone_masks)
        log_duration_targets = (
            torch.log(inputs["durations"].float() + 1)
            .detach()
            .masked_select(phone_masks)
        )
        egemap_targets = inputs["egemap_features"]

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        losses_dict = {
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "duration_loss": duration_loss,
        }

        if egemap_targets is not None:
            egemap_loss = self.mse_loss(egemap_predictions, egemap_targets)
            losses_dict["egemap_loss"] = egemap_loss
        else:
            egemap_loss = torch.FloatTensor([0]).detach().to(device)

        if not compute_mel_loss:
            return losses_dict

        mel_predictions = mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )  # b, t, 1 -> b, t, c
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss + egemap_loss
        losses_dict["total_loss"] = total_loss
        losses_dict["mel_loss"] = mel_loss

        return losses_dict
