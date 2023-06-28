import torch
import torch.nn as nn


class JCU(nn.Module):
    def __init__(self, config):
        super(JCU, self).__init__()
        self.unconditional_convs = nn.ModuleList()
        embedding_size = config.emb_size_dis
        input_channel = config.n_mel_channels
        kernels = config.kernels_d
        strides = config.strides_d

        for d, kernel, stride in zip([8, 4, 1, 4, embedding_size], kernels, strides):
            out_channel = int(embedding_size / d)
            self.unconditional_convs.extend(
                [
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            input_channel,
                            out_channel,
                            kernel_size=kernel,
                            stride=stride,
                        )
                    ),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            input_channel = out_channel

        self.fc = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), nn.LeakyReLU(0.2, True)
        )

        self.conditional_convs = nn.ModuleList()
        input_channel = embedding_size
        for d, kernel, stride in zip([4, embedding_size], kernels[3:], strides[3:]):
            out_channel = int(embedding_size / d)
            self.conditional_convs.extend(
                [
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            input_channel,
                            out_channel,
                            kernel_size=kernel,
                            stride=stride,
                        )
                    ),
                    nn.LeakyReLU(0.2, True),
                ]
            )
            input_channel = out_channel

    def forward(self, x, embedding):
        fmaps = []
        for i in range(0, 6):
            x = self.unconditional_convs[i](x)
            if i % 2 == 0:
                fmaps.append(x)
        conditional_out = self.fc(embedding)
        conditional_out = conditional_out.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        conditional_out = torch.cat([x, conditional_out], dim=2)
        for i, layer in enumerate(self.conditional_convs):
            conditional_out = layer(conditional_out)
            if i % 2 == 0:
                fmaps.append(conditional_out)
        unconditional_out = x
        for i in range(6, 10):
            unconditional_out = self.unconditional_convs[i](unconditional_out)
            if i % 2 == 0:
                fmaps.append(unconditional_out)

        return unconditional_out.squeeze(1), conditional_out.squeeze(1), fmaps
