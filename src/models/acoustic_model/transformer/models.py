import json
import torch
import numpy as np
import torch.nn as nn

from src.models.acoustic_model.transformer.layers import FFTBlock


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.padding_index = config.padding_index
        n_position = config.max_seq_len + 1
        with open(config.phones_mapping_path) as f:
            phones_dict = json.load(f)
        n_src_vocab = len(phones_dict.keys()) + 1
        d_word_vec = config.transformer_encoder_hidden
        n_layers = config.transformer_encoder_layer
        n_head = config.transformer_encoder_head
        d_k = d_v = config.transformer_encoder_hidden // config.transformer_encoder_head
        d_model = config.transformer_encoder_hidden
        d_inner = config.transformer_conv_filter_size
        kernel_size = config.transformer_conv_kernel_size
        dropout = config.transformer_encoder_dropout

        self.max_seq_len = config.max_seq_len
        self.d_model = d_model
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=config.padding_index
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                n_position, d_word_vec, padding_idx=self.padding_index
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_inner,
                    kernel_size,
                    dropout=dropout,
                    conditional_layer_norm=config.conditional_layer_norm,
                    conditional_cross_attention=config.conditional_cross_attention
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, speaker_emotion_embedding=None):
        """
        encoder output: bs, seq_len, hid
        att_outputs: list[bs * n_head, seq_len], len = n_layers
                     att_outputs[0] – output from first transformer layer
                     att_outputs[0][0] – output from 1 transformer layer, 1 head, 1 sample in batch
                     att_outputs[0][bs] – output from 1 transformer layer, 2 head, 1 sample in batch
        """
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        attention_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        att_outputs = []

        if not self.training and src_seq.shape[1] > self.max_seq_len:
            sinusoid_encoding_table = get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model, padding_idx=self.padding_index
            )[: src_seq.shape[1], :]
            sinusoid_encoding_table = (
                sinusoid_encoding_table.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .to(src_seq.device)
            )
            encoder_output = self.src_word_emb(src_seq) + sinusoid_encoding_table
        else:
            encoder_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
        for i, encoder_layer in enumerate(self.layer_stack):
            encoder_output, attention_weights = encoder_layer(
                encoder_output,
                mask=mask,
                speaker_emotion_embedding=speaker_emotion_embedding,
                attention_mask=attention_mask,
            )
            att_outputs.append(attention_weights)
        return encoder_output, att_outputs


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.padding_index = config.padding_index
        n_position = config.max_seq_len + 1
        d_word_vec = config.transformer_decoder_hidden
        n_layers = config.transformer_decoder_layer
        n_head = config.transformer_decoder_head
        d_k = config.transformer_decoder_hidden // config.transformer_decoder_head
        d_model = config.transformer_decoder_hidden
        d_inner = config.transformer_conv_filter_size
        kernel_size = config.transformer_conv_kernel_size
        dropout = config.transformer_decoder_dropout
        self.max_seq_len = config.max_seq_len
        self.d_model = d_model
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(
                n_position, d_word_vec, padding_idx=self.padding_index
            ).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model,
                    n_head,
                    d_k,
                    d_inner,
                    kernel_size,
                    dropout=dropout,
                    conditional_layer_norm=config.conditional_layer_norm,
                    conditional_cross_attention=config.conditional_cross_attention
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, encoded_sequence, mask, speaker_emotion_embedding=None):
        """
        decoder output: bs, mel_len, hid
        att_outputs: list[bs * n_head, mel_len], len = n_layers
                     att_outputs[0] – output from first transformer layer
                     att_outputs[0][0] – output from 1 transformer layer, 1 head, 1 sample in batch
                     att_outputs[0][bs] – output from 1 transformer layer, 2 head, 1 sample in batch
        """

        batch_size, max_len = encoded_sequence.shape[0], encoded_sequence.shape[1]
        att_outputs = []

        if not self.training and encoded_sequence.shape[1] > self.max_seq_len:
            attention_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            sinusoid_encoding_table = get_sinusoid_encoding_table(
                encoded_sequence.shape[1], self.d_model, padding_idx=self.padding_index
            )
            sinusoid_encoding_table = sinusoid_encoding_table[
                : encoded_sequence.shape[1], :
            ]
            sinusoid_encoding_table = sinusoid_encoding_table.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            sinusoid_encoding_table = sinusoid_encoding_table.to(
                encoded_sequence.device
            )
            decoder_output = encoded_sequence + sinusoid_encoding_table
        else:
            max_len = min(max_len, self.max_seq_len)
            attention_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            decoder_output = encoded_sequence[:, :max_len, :]
            decoder_output = decoder_output + self.position_enc[:, :max_len, :].expand(
                batch_size, -1, -1
            )
            mask = mask[:, :max_len]
            attention_mask = attention_mask[:, :, :max_len]
        for decoder_layer in self.layer_stack:
            decoder_output, attention_output = decoder_layer(
                decoder_output,
                mask=mask,
                speaker_emotion_embedding=speaker_emotion_embedding,
                attention_mask=attention_mask,
            )
            att_outputs.append(attention_output)

        return decoder_output, mask, att_outputs
