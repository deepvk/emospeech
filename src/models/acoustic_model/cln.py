import torch
import torch.nn as nn


class ConditionalLayerNorm(nn.Module):
    def __init__(self, normal_shape: int, epsilon=1e-5):
        super(ConditionalLayerNorm, self).__init__()
        self.epsilon = epsilon
        self.W_scale = nn.Linear(normal_shape, normal_shape)
        self.W_bias = nn.Linear(normal_shape, normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x: torch.Tensor, emotion_embedding: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(emotion_embedding)
        bias = self.W_bias(emotion_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)

        return y
