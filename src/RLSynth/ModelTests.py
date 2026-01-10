import numpy as np
import Models as M
import torch
import torch.nn as nn
import json
import Environment
import wandb
import math
import matplotlib.pyplot as plt


class RegModel(nn.Module):
    def __init__(self, m_type='2D'):
        super(RegModel, self).__init__()
        sizes = [3 for i in range(3)]
        n = [24, 48, 64]
        # n = [24, 24, 24, 40, 40, 40, 56, 56, 56, 56]
        if m_type == '2D':
            self.base = M.SimpleCNN(n, sizes, 1000)
        elif m_type == '1D':
            self.base = M.SimpleCNN1D(n, sizes, 1000)
        self.layers = nn.ModuleList([nn.Linear(1000, 500)])  # List of layers
        nn.init.normal_(self.layers[0].weight, 0, 2 / np.sqrt(1000))

        self.layers.extend([nn.Linear(500, 500) for i in range(3 - 1)])
        for layer in self.layers[1:]:
            nn.init.normal_(layer.weight, 0, 2 / np.sqrt(500))

        self.final = nn.Linear(500, 1)
        nn.init.normal_(self.final.weight, 0, 2 / np.sqrt(1000))

        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.base(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.final(x)
        x = self.sigmoid(x) * 4186
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            n_odd = int((d_model + 1) / 2)
            n_even = int(d_model / 2)
            position_odd = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_odd).t()
            position_even = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_even).t()
        else:
            n_odd = n_even = int(d_model / 2)
            position_odd = position_even = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_even).t()
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(1).repeat(1, max_len)
        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2] = torch.sin(position_odd * div_term[:n_odd])
        pe[0, 1::2] = torch.cos(position_even * div_term[:n_even])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


env, agent, critic = Environment.initialize_env(config_dir=r'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Level1_Synth.json',
                                                training_config_dir=r'Training_Config_Parameters.json')
model = RegModel(m_type='2D')
# model = M.AttentionModel(max_len=env.desired_signal_mel_spec.shape[2],
#                          n_mels=env.desired_signal_mel_spec.shape[1],
#                          model_dim=400)
model = model.to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=1e-6)

freq_pos_code = PositionalEncoding(d_model=env.desired_signal_mel_spec.shape[1],
                                   max_len=env.desired_signal_mel_spec.shape[2]).to('cuda')

time_pos_code = PositionalEncoding(d_model=env.desired_signal_mel_spec.shape[2],
                                   max_len=env.desired_signal_mel_spec.shape[1]).to('cuda')

wandb.init(project='Synth_with_RL')
for i in range(4000):
    avg_loss = 0
    batch_n = 0
    for e in range(100):
        curr_mel = env.desired_signal_mel_spec
        curr_mel = freq_pos_code(curr_mel)
        curr_mel = time_pos_code(curr_mel.transpose(1, 2))
        plt.imshow(curr_mel.cpu().squeeze(0).numpy())
        plt.show()
        curr_signal = env.desired_signal
        curr_freq = env.synth_to_copy.oscillators[0].freq
        out_freq = model(curr_mel)
        # out_freq = model(curr_signal.unsqueeze(0))
        loss = torch.pow(out_freq - curr_freq, 2)
        loss.backward()
        avg_loss += loss.item()
        env.randomize_synth()
        if batch_n == 20:
            opt.step()
            opt.zero_grad()
            batch_n = 0
        batch_n += 1

    wandb.log({"loss": avg_loss / 100})
wandb.finish()
