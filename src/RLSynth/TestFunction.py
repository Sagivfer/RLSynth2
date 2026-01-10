import numpy as np
import Models as M
import torch
import torch.nn as nn
import json
import Environment
import wandb
import math
import matplotlib.pyplot as plt


class Stam(nn.Module):
    def __init__(self, n_parameters, model_dim, n_layers):
        super(Stam, self).__init__()
        initialization_std = np.sqrt(2 / model_dim)

        self.n_parameters = n_parameters
        self.layers = nn.ModuleList([nn.Linear(2, model_dim)])  # List of layers
        nn.init.normal_(self.layers[0].weight, 0, initialization_std)

        self.layers.extend([nn.Linear(model_dim, model_dim) for i in
                            range(n_layers - 1)])  # Extending the list of layers with a list of layers
        for layer in self.layers[1:]:
            nn.init.normal_(layer.weight, 0, initialization_std)

        self.activation = nn.ReLU()

        self.mu = nn.Linear(model_dim, self.n_parameters)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        value = self.mu(x)
        value = torch.sigmoid(value)
        point = value * 5

        return point


model = Stam()
model = model.to('cuda')
opt = torch.optim.Adam(model.parameters(), lr=1e-6)

wandb.init(project='Synth_with_RL')
for epoch in range(10000):
    avg_loss = 0
    batch_n = 0
    for mini_batch in range(100):
        curr_mel = env.desired_signal_mel_spec
        curr_freq = env.synth_to_copy.oscillators[0].freq.value / 4186
        curr_amp = env.synth_to_copy.oscillators[0].amp.value
        parameters = model(curr_mel)

        loss = torch.pow(parameters[0] - curr_freq, 2) + torch.pow(parameters[1] - curr_amp, 2)
        loss.backward()
        avg_loss += loss.item()
        env.randomize_synth(randomize_type=2)
        if batch_n == 20:
            opt.step()
            opt.zero_grad()
            batch_n = 0
        batch_n += 1

    wandb.log({"loss": avg_loss / 100})
    if (avg_loss / 100) <= 0.00001:
        break
wandb.finish()
