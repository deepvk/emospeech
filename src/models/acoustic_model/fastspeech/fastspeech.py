import torch
import torch.nn as nn

from src.models.acoustic_model.fastspeech.modules import VarianceAdaptor
from src.models.acoustic_model.transformer.models import Decoder, Encoder
from src.utils.fastspeech_utils import get_mask_from_lengths
from src.utils.utils import crash_with_msg


class FastSpeech2(nn.Module):
    def __init__(self, config):
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.n_emotions = config.n_emotions
        self.variance_adaptor = VarianceAdaptor(config)
        self.emotion_emb = nn.Embedding(
            config.n_emotions, config.emotion_emb_hidden_size
        )
        self.mel_linear = nn.Linear(
            config.transformer_decoder_hidden, config.n_mel_channels
        )
        self.speaker_emb = nn.Embedding(
            config.n_speakers + 1, config.speaker_emb_hidden_size
        )

        # Advanced Emotion Conditioning
        self.conditional_cross_attention = config.conditional_cross_attention
        self.conditional_layer_norm_usage = config.conditional_layer_norm
        self.stack_speaker_with_emotion_embedding = (
            config.stack_speaker_with_emotion_embedding
        )

    def forward(self, device, batch_dict) -> dict:
        src_masks = get_mask_from_lengths(batch_dict["text_lens"], device)
        mel_masks = (
            get_mask_from_lengths(batch_dict["mel_lens"], device)
            if batch_dict["mel_lens"] is not None
            else None
        )

        emotion_embedding = self.emotion_emb(batch_dict["emotions"].to(device))
        speaker_embedding = self.speaker_emb(batch_dict["speakers"].to(device))

        if self.stack_speaker_with_emotion_embedding:
            emotion_embedding = torch.hstack([emotion_embedding, speaker_embedding])

        if self.conditional_cross_attention or self.conditional_layer_norm_usage:
            encoder_output, encoder_attention = self.encoder(
                batch_dict["texts"].to(device),
                src_masks.to(device),
                speaker_emotion_embedding=emotion_embedding,
            )
        else:
            encoder_output, encoder_attention = self.encoder(
                batch_dict["texts"].to(device),
                src_masks.to(device),
                speaker_emotion_embedding=None,
            )

        if not self.stack_speaker_with_emotion_embedding:
            max_src_len = torch.max(batch_dict["text_lens"]).item()
            encoder_output = (
                encoder_output
                + speaker_embedding.unsqueeze(1).expand(-1, max_src_len, -1)
                + emotion_embedding.unsqueeze(1).expand(-1, max_src_len, -1)
            )

        if not self.conditional_cross_attention:
            max_src_len = torch.max(batch_dict["text_lens"]).item()
            encoder_output = encoder_output + emotion_embedding.unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        var_adaptor_output = self.variance_adaptor(
            device, encoder_output, src_masks, batch_dict
        )
        mel_lens = (
            batch_dict["mel_lens"]
            if batch_dict["mel_lens"] is not None
            else var_adaptor_output["mel_len"]
        )
        mel_masks = (
            mel_masks
            if mel_masks is not None
            else get_mask_from_lengths(mel_lens, device=device)
        )
        if self.conditional_cross_attention or self.conditional_layer_norm_usage:
            output, mel_masks, decoder_attention = self.decoder(
                var_adaptor_output["output"],
                mel_masks,
                speaker_emotion_embedding=emotion_embedding,
            )
        else:
            output, mel_masks, decoder_attention = self.decoder(
                var_adaptor_output["output"], mel_masks, speaker_emotion_embedding=None
            )
        output = self.mel_linear(output)

        if batch_dict["mels"] is not None:
            if output.shape != batch_dict["mels"].shape:
                msg = (
                    f"Expected Variational Adapter Output to be equal to the target mel "
                    f"found target: {batch_dict['mels'].shape}, output: {output.shape}."
                )
                crash_with_msg(msg)

        output_dict = {
            "predicted_mel": output,
            "predicted_pitch": var_adaptor_output["pitch_prediction"],
            "predicted_energy": var_adaptor_output["energy_prediction"],
            "predicted_egemap": var_adaptor_output["egemap_prediction"],
            "predicted_log_durations": var_adaptor_output["log_duration_prediction"],
            "predicted_durations_rounded": var_adaptor_output["duration_rounded"],
            "mel_len": var_adaptor_output["mel_len"],
            "phone_masks": src_masks,
            "mel_masks": mel_masks,
            "emotion_embedding": emotion_embedding,
            "encoder_attention": encoder_attention,
            "decoder_attention": decoder_attention,
        }

        return output_dict
