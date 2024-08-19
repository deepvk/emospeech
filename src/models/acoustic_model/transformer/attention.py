import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.acoustic_model.cln import ConditionalLayerNorm


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, cca: False, mask: torch.Tensor = None):
        """
        For Self-Attention:
             q, k, v: encoder input  bs * n_head, seg_len, hid
        For CCA:
             q: encoder input bs * n_head, seg_len, hid
             k, v: emotion embedding  bs, 1, hid

        return: bs * n_head, seg_len, hid
        """
        bs, seq_len, hid = q.size()  # bs * n_head, seg_len, hid

        attention_weights = torch.bmm(q, k.transpose(1, 2)).squeeze(-1)
        attention_weights = attention_weights / self.scale

        if mask is not None:
            if mask.shape == attention_weights.shape:
                attention_weights = attention_weights.masked_fill(
                    mask, torch.finfo(q.dtype).min
                )
            else:
                attention_weights = attention_weights.masked_fill(
                    mask[:, 0, :], torch.finfo(q.dtype).min
                )

        if cca:
            attention_weights = F.softmax(attention_weights, dim=1)
            context_vector = torch.bmm(attention_weights.unsqueeze(-1), v)
        else:
            attention_weights = F.softmax(attention_weights, dim=2)
            context_vector = torch.bmm(attention_weights, v).squeeze(1)
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, conditional_layer_norm, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_k)
        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
        self.layer_norm = (
            ConditionalLayerNorm(d_model)
            if conditional_layer_norm
            else nn.LayerNorm(d_model)
        )
        self.fc = nn.Linear(n_head * d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q, k, v, speaker_emotion_embedding=None, mask=None, cca: bool = False
    ):
        # q: encoder output [bs, max_seq_len, hid]
        # k:  encoder output [bs, max_seq_len, hid] / emotion_embedding [bs, hid]
        # v:  encoder output [bs, max_seq_len, hid] / emotion_embedding [bs, hid]
        if len(k.size()) == 2:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        bs, seq_len, hidden_size = q.size()
        kv_len = k.size()[1]
        residual = q
        q = self.w_qs(q).view(bs, seq_len, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, kv_len, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, kv_len, self.n_head, self.d_k)
        q = (
            q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, self.d_k)
        )  # (n*b) x lq x dk
        k = (
            k.permute(2, 0, 1, 3).contiguous().view(-1, kv_len, self.d_k)
        )  # (n*b) x 1 x dk
        v = (
            v.permute(2, 0, 1, 3).contiguous().view(-1, kv_len, self.d_k)
        )  # (n*b) x 1 x dv
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1)
        context_vector, attention_weights = self.attention(q, k, v, cca=cca, mask=mask)

        context_vector = context_vector.view(self.n_head, bs, seq_len, self.d_k)
        context_vector = (
            context_vector.permute(1, 2, 0, 3).contiguous().view(bs, seq_len, -1)
        )  # b x lq x (n*dk)
        context_vector = self.dropout(self.fc(context_vector))
        if speaker_emotion_embedding is not None:
            context_vector = self.layer_norm(
                context_vector + residual, speaker_emotion_embedding
            )
        else:
            context_vector = self.layer_norm(context_vector + residual)

        return context_vector, attention_weights
