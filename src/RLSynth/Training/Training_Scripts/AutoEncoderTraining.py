import numpy as np
import RL_Synth.Models as M
import SynthModules as SM
import torch
import torch.nn as nn
import json
import src.RLSynth.Environment as Environment
import wandb


with open(r'../Training Configs/EncoderDecoderTraining_Config.json') as f:
    training_config = dict(json.load(f))
    representation_dim = training_config["representation_dim"]
    device = training_config["device"]
    n_steps = training_config["n_steps"]
    T = training_config["episode_len"]
    lr_agent = training_config["lr_agent"]
    lr_critic = training_config["lr_critic"]
    baseline = training_config["baseline"]
    algorithm = training_config["algorithm"]
    experiment_num = training_config["experiment_num"]
    training_config["experiment_num"] += 1
    batch_size = training_config["batch_size"]

with open(r'../../../Dynamic_synth/Synth Configs/EncoderDecoderSynth.json') as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]

env, agent, critic = Environment.initialize_env(config_dir=r'../../../Dynamic_synth/Synth Configs/EncoderDecoderSynth.json',
                                                training_config_dir=r'../Training Configs/EncoderDecoderTraining_Config.json',
                                                training_type=2)
model = M.EncoderDecoder([24, 48, 64], [3, 3, 3], representation_dim, env.n_frames)
model = model.to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_agent)
file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\EncoderDecoder.pt"

counter = 0
loss = 0
wandb.init(
    project='Synth_with_RL',
    config=training_config
)

for e in range(n_steps[0]):
    print(e)
    input = env.desired_signal_mel_spec
    output = model(input)
    curr_loss = loss_func(input, output) / batch_size
    if not torch.isnan(curr_loss).item():
        loss = curr_loss + loss
        counter += 1

    if counter == batch_size:

        wandb.log({
            'Reconstruction Error': loss,
            })

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = 0
        counter = 1

    env.randomize_synth(randomize_type=2)

torch.save(model.state_dict(), file_dir)
