import Dynamic_synth.SynthModules as SM
import torch
import json
import matplotlib.pyplot as plt
from RL_Synth import Models, Environment, Algorithms
import numpy as np
import torch.nn as nn


with open(r'Training Configs/Training_Config.json') as f:
    training_config = dict(json.load(f))
    representation_dim = training_config["representation_dim"]
    device = training_config["device"]
    n_steps = training_config["n_steps"]
    T = training_config["episode_len"]
    lr_agent = training_config["lr_agent"]
    lr_critic = training_config["lr_critic"]
    baseline = training_config["baseline"]
    randomize_time = training_config["randomize_time"]
    n_layers = training_config["n_layers"]
    algorithm = training_config["algorithm"]
    batch_size = training_config["batch_size"]
    experiment_num = training_config["experiment_num"] - 1

with open(r'OscillatorFreq.json') as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config)

t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), device=device)
env, agent, critic = Environment.initialize_env(config_dir=r'OscillatorFreq.json',
                                                training_config_dir=r'Training Configs/Training_Config.json')
plt.imshow(env.desired_signal_mel_spec.cpu().squeeze(0).numpy())
plt.show()
n_levels = 2

for i in range(2, n_levels + 1):
    with open(fr'Level{i}_Synth.json') as f:
        synth_config2 = dict(json.load(f))
        agent.level_up(synth_config2)

file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\{algorithm}\{experiment_num}"

agent.load_models(fr"{file_dir}\Agent.pt")

for i, p in enumerate(agent.Backbone.melspec_features_model.convs[0].named_parameters()):
    print(p[1].shape)
    for j in range(10):
        feature = p[1][j][0].detach().cpu().numpy()
        print(feature)

        pic = (255 * (feature - np.min(feature))/np.ptp(feature)).astype(int)
        plt.imshow(feature)
        plt.show()
    break

for i, p in enumerate(agent.Backbone.waveform_features_model.convs[0].named_parameters()):
    print(p[1].shape)
    for j in range(64):
        feature = p[1][j].detach().cpu().numpy()
        print(feature)

        pic = (255 * (feature - np.min(feature))/np.ptp(feature)).astype(int)
        plt.imshow(feature)
        plt.show()
    break

with torch.no_grad():
    env.randomize_synth()
    # print(agent.Backbone.fc.weight.shape)
    # print(agent.Backbone(env.mel_spec(env.synth(t)).unsqueeze(0)).cpu().shape)

