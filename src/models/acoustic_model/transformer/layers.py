import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.acoustic_model.transformer.attention import (
    MultiHeadAttention, ScaledDotProductAttention)
from src.utils.utils import crash_with_msg


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2
        )
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2
        )
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emotion_cross_attention_output=None):
        residual = x
        if emotion_cross_attention_output is not None:
            output = self.layer_norm(x + emotion_cross_attention_output)
        else:
            output = self.layer_norm(x)
        output = output.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_k,
        d_inner,
        kernel_size,
        dropout=0.1,
        conditional_layer_norm=False,
        conditional_cross_attention=False,
        separate_head=False,
    ):
        super(FFTBlock, self).__init__()
        self.mha = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            dropout=dropout,
            conditional_layer_norm=conditional_layer_norm,
        )
        self.pos_ffn = PositionWiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )
        self.conditional_cross_attention = conditional_cross_attention
        if conditional_cross_attention:
            self.cca = MultiHeadAttention(
                n_head,
                d_model,
                d_k,
                dropout=dropout,
                conditional_layer_norm=conditional_layer_norm,
            )
        else:
            self.cca = None

    def forward(
        self, enc_input, mask=None, attention_mask=None, speaker_emotion_embedding=None
    ):
        self_attention_output, _ = self.mha(
            q=enc_input,
            k=enc_input,
            v=enc_input,
            cca=False,
            speaker_emotion_embedding=speaker_emotion_embedding,
            mask=attention_mask,
        )
        self_attention_output = self_attention_output.masked_fill(mask.unsqueeze(-1), 0)
        if self.cca is not None:
            emotion_cross_attention_output, attention_weights = self.cca(
                q=enc_input,
                k=speaker_emotion_embedding,
                v=speaker_emotion_embedding,
                cca=True,
                speaker_emotion_embedding=speaker_emotion_embedding,
                mask=attention_mask,
            )

            emotion_cross_attention_output = emotion_cross_attention_output.masked_fill(
                mask.unsqueeze(-1), 0
            )
        else:
            emotion_cross_attention_output = None
            attention_weights = None
        ffn_output = self.pos_ffn(
            self_attention_output,
            emotion_cross_attention_output=emotion_cross_attention_output,
        )
        ffn_output = ffn_output.masked_fill(mask.unsqueeze(-1), 0)

        return ffn_output, attention_weights


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            if kernel_size % 2 != 1:
                crash_with_msg(
                    f"Kernel size of CovNorm is: {kernel_size}, supposed to be %2 == 0"
                )
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal
