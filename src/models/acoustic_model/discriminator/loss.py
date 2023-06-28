import torch
import torch.nn as nn

from src.models.acoustic_model.discriminator.jcu_discriminator import JCU


class AdversarialLoss(nn.Module):
    def __init__(self, config):
        super(AdversarialLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.n_mels = config.n_mel_channels
        self.fm_loss = config.compute_fm_loss

    def _feature_matching_loss(self, generated_fmaps, gt_fmaps):
        res = []
        for generated_fm, gt_fm in zip(generated_fmaps, gt_fmaps):
            res.append(torch.mean(self.l1_loss(gt_fm, generated_fm)))
        return torch.mean(torch.tensor(res))

    def generator_loss(self, inputs: dict, predictions: dict, jcu: JCU):
        mel_predictions = predictions["predicted_mel"]
        emotion_embedding = predictions["emotion_embedding"]
        gen_conditional_out, gen_unconditional_out, gen_feature_maps = jcu(
            mel_predictions.transpose(1, 2), emotion_embedding
        )

        if self.fm_loss:
            mel_targets = inputs["mels"].detach()
            _, _, gt_feature_maps = jcu(mel_targets.transpose(1, 2), emotion_embedding)
            fm_loss = self._feature_matching_loss(gen_feature_maps, gt_feature_maps)
        else:
            fm_loss = torch.zeros(1, requires_grad=False).to(gen_conditional_out.device)

        generator_loss = 0.5 * (
            self.mse_loss(gen_conditional_out, torch.ones_like(gen_conditional_out))
            + self.mse_loss(
                gen_unconditional_out, torch.ones_like(gen_unconditional_out)
            )
        )

        return generator_loss, fm_loss

    def discriminator_loss(self, inputs: dict, predictions: dict, jcu: JCU):
        mel_targets = inputs["mels"]
        emotion_embedding = predictions["emotion_embedding"].detach()

        gt_conditional_out, gt_unconditional_out, _ = jcu(
            mel_targets.transpose(1, 2), emotion_embedding
        )

        discriminator_loss_real = 0.5 * (
            self.mse_loss(gt_conditional_out, torch.ones_like(gt_conditional_out))
            + self.mse_loss(gt_unconditional_out, torch.ones_like(gt_unconditional_out))
        )

        mel_predictions = predictions["predicted_mel"].detach()
        gen_conditional_out, gen_unconditional_out, _ = jcu(
            mel_predictions.transpose(1, 2), emotion_embedding
        )
        discriminator_loss_fake = 0.5 * (
            torch.mean(gen_conditional_out**2)
            + torch.mean(gen_unconditional_out**2)
        )

        return discriminator_loss_real + discriminator_loss_fake
