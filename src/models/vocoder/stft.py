"""
BSD 3-Clause License
Copyright (c) 2017, Prem Seetharaman
All rights reserved.
* Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Tuple

import numpy as np
import torch
from scipy.signal import get_window


class TorchSTFT(torch.nn.Module):
    def __init__(
        self, gen_istft_n_fft: int, gen_istft_hop_size: int, window="hann", **_
    ):
        super().__init__()
        self.filter_length = gen_istft_n_fft
        self.hop_length = gen_istft_hop_size
        self.win_length = gen_istft_n_fft
        self.window = torch.from_numpy(
            get_window(window, gen_istft_n_fft, fftbins=True).astype(np.float32)
        )

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.complex]:
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.complex) -> torch.Tensor:
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )

        return inverse_transform.unsqueeze(
            -2
        )  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)

        return reconstruction
