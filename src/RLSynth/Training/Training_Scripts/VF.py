import matplotlib.pyplot as plt

import Dynamic_synth.SynthModules as SM
import torch
import json
from RL_Synth import Models, Environment, Algorithms, Utils
from utils import metrics
import os
import wandb
import numpy as np

with open(r'../Training_Configs/Training_Config.json') as f:
    training_config = dict(json.load(f))
    representation_dim = training_config["representation_dim"]
    device = training_config["device"]
    n_steps = training_config["n_steps"]
    T = training_config["episode_len"]
    lr_agent = training_config["lr_agent"]
    lr_critic = training_config["lr_critic"]
    lr_representation = training_config["lr_representation"]
    baseline = training_config["baseline"]
    algorithm = training_config["algorithm"]
    experiment_num = training_config["experiment_num"]
    training_config["experiment_num"] += 1
    batch_size = training_config["batch_size"]

synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreqWave.json'

with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

# for lr_rep in [1e-04, 5e-05, 1e-05, 5e-06]:
#     for lr_crit in [1e-03, 1e-04, 1e-05, 1e-06]:
env, agent, critic = Environment.initialize_env(config_dir=synth_file,
                                                training_config_dir=r'../Training_Configs/Training_Config.json')

file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\VF"

try:
    os.mkdir(file_dir)
except:
    pass

with open(fr'{file_dir}\config.json', 'w') as f:
    json.dump(training_config, f)

with open(fr'../Training_Configs/Training_Config.json', 'wt') as f:
    json.dump(training_config, f)

agent.to('cuda')
critic.to('cuda')

critic_models, critic_model_names = critic.get_models(models=[], names=[])
critic_param_list = list()
for model, name in zip(critic_models, critic_model_names):
    if name != 'SoundEncoder':
        for p_name, param in model.named_parameters(recurse=True):
            param_dict = {'params': param, 'lr': lr_critic}
            critic_param_list.append(param_dict)
critic.model_parameters = critic_param_list

opt = torch.optim.RMSprop(critic_param_list, weight_decay=0.001)
opt.add_param_group({'params': agent.SoundEncoder.parameters(), 'lr': lr_representation})
opt.add_param_group({'params': agent.advanced_representation.parameters(), 'lr': lr_representation})

warmup_scheduler = Utils.GradualWarmupScheduler(opt, total_steps=1000)
wandb.init(project='Synth_with_RL')

count_batch = 0
loss = 0
rep_size = 0
magnitude = 0

for t in range(1000000):
    print(t)
    count_batch += 1
    env.randomize_synth(randomize_type=3)
    env.update_signal(update_type=3)
    env_value = env.get_value()

    state = env.get_state()

    state_representation, _ = agent.get_state_representation(state)

    critic_value = critic.get_value_of_state(state_representation)

    # print(critic_value, value)
    this_loss = torch.abs(critic_value - env_value)
    loss = loss + this_loss
    rep_size = rep_size + this_rep_size

    # if t % 100 == 0 and t > 1000:
    #     wave1 = env.synth.oscillators[0].waveform.value
    #     wave2 = env.synth.oscillators[1].waveform.value
    #     wandb.log({'freq1': env.synth.oscillators[0].freq.value,
    #                'freq2': env.synth.oscillators[1].freq.value,
    #                'wave1': env.synth.oscillators[0].waveform.option2idx[wave1],
    #                'wave2': env.synth.oscillators[0].waveform.option2idx[wave2],
    #                'loss_t': this_loss.item()
    #                })
    if count_batch == batch_size:
        loss = loss / batch_size
        rep_size = rep_size / batch_size
        loss.backward()
        opt.step()
        opt.zero_grad()
        warmup_scheduler.step()
        # if t > 1000:
        wandb.log({"loss": loss,
                   "rep_size": rep_size})
        loss = 0
        count_batch = 0

wandb.finish()


cnn_model_path = fr'{file_dir}\cnn_vf_pretrain.pth'
mlp_model_path = fr'{file_dir}\mlp_vf_pretrain.pth'
torch.save(agent.SoundEncoder.state_dict(), cnn_model_path)
torch.save(agent.advanced_representation.state_dict(), mlp_model_path)
