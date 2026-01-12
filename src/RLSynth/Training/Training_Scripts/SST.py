"""
Shared Space Training
"""

import Dynamic_synth.SynthModules as SM
import torch
import json
from src.RLSynth import Models, Environment, Algorithms
from torch import nn
import os
import wandb
import numpy as np


def get_random_agents(top_agent):
    current_agent = top_agent
    agents_indices = list()
    while len(current_agent.next_agents) > 0:
        next_agent_index = np.random.choice(range(len(current_agent.next_agents)))
        agents_indices.append(torch.tensor(next_agent_index))
        current_agent = current_agent.next_agents[next_agent_index]
    return current_agent, agents_indices


with open(r'../Training Configs/Training_Config.json') as f:
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

with open(fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Level4_Synth.json') as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

env, agent, critic = Environment.initialize_env(config_dir=r'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Level4_Synth.json',
                                                training_config_dir=r'../Training Configs/Training_Config.json')
file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\SST"
try:
    os.mkdir(file_dir)
except:
    pass

with open(fr'{file_dir}\config.json', 'w') as f:
    json.dump(training_config, f)

with open(fr'../Training Configs/Training_Config.json', 'wt') as f:
    json.dump(training_config, f)

agent.to('cuda')
critic.to('cuda')
agent_param_list = list()
# Building the parameter list for the optimizer.
# Stopping criteria has lower learning rate
level_to_model = Environment.DFS_agent_parameters(agent)
for level, agents_models in level_to_model.items():
    for model, sub_agent in agents_models:
        for sub_model_name, sub_model in model.named_children():
            if sub_model_name == 'stopping_criteria':
                lr = lr_agent * (10 ** level)
            else:
                lr = lr_agent * (10 ** (level + 1))
            param_dict = {'params': list(sub_model.parameters()), 'lr': lr}
            agent_param_list.append(param_dict)
agent.model_parameters = agent_param_list

critic_models, critic_model_names = critic.get_models(models=[], names=[])
critic_param_list = list()
for model, name in zip(critic_models, critic_model_names):
    if name != 'SoundEncoder' and name != 'ParametersEncoder':
        for p_name, param in model.named_parameters(recurse=True):
            param_dict = {'params': param, 'lr': lr_critic}
            critic_param_list.append(param_dict)
critic.model_parameters = critic_param_list

encoding2spec = Models.SimpleCNNDecoder([64, 48, 24],
                                        [3, 3, 3],
                                        representation_dim,
                                        env.n_frames).to(torch.float32).to(device)

opt = torch.optim.Adam(agent_param_list + critic_param_list, weight_decay=0.05)
opt.add_param_group({'params': agent.SoundEncoder.parameters(), 'lr': 1e-5})
opt.add_param_group({'params': agent.ParametersEncoder.parameters(), 'lr': 1e-5})
opt.add_param_group({'params': encoding2spec.parameters(), 'lr': 1e-5})

wandb.init(project='Synth_with_RL')
count_batch = 0
count_test = 0

mse = nn.MSELoss()
avg_loss = 0
loss_parameters = 0
loss_reconstruction = 0
loss_representation_diff = 0
loss = 0
magnitude = 0

update_diversity = False
prev_for_diversity = None

for t in range(1000000):
    print(t)

    env.randomize_synth(randomize_type=1)
    current_agent, agent_indices = get_random_agents(agent)

    count_batch += 1
    count_test += 1

    action_temp = list(current_agent.get_random_action())
    action = agent_indices + action_temp

    prev_signal_representation = agent.get_sound_features(env.current_signal)
    prev_synth_representation = agent.get_synth_features(env.curr_synth_parameters)
    prev_mel_spec = env.current_signal_mel_spec
    prev_synth_parameters = env.curr_synth_parameters

    # changed_module = env.synth_modules[action[0]]
    # parameter = changed_module.available_parameters[action[1]]
    # if current_agent.regression is not None:
    #     parameter_value = changed_module.__getattribute__(parameter)
    #     future_value = parameter_value + action[2].item()
    #     if future_value < current_agent.min_val:
    #         action[2] = torch.tensor([current_agent.min_val - parameter_value], device=device)
    #     if future_value > current_agent.max_val:
    #         action[2] = torch.tensor([current_agent.max_val - parameter_value], device=device)
    #
    # env.step(action=action)

    # current_signal_representation = agent.get_sound_features(env.current_signal)
    # current_synth_representation = agent.get_synth_features(env.curr_synth_parameters)
    # current_mel_spec = env.current_signal_mel_spec
    # current_synth_parameters = env.curr_synth_parameters

    extremity_loss_synth = torch.mean(torch.abs(prev_synth_representation))
    extremity_loss_sound = torch.mean(torch.abs(prev_signal_representation))

    parameters_regression_loss = agent.get_parameters_synth_loss(prev_signal_representation)
    reconstruction = encoding2spec(prev_synth_representation)
    reconstruction_loss = mse(reconstruction, prev_mel_spec)

    representation_diff_loss = mse(prev_synth_representation, prev_signal_representation.detach())

    extremity_loss = 0.5 * extremity_loss_sound + 6 * extremity_loss_synth
    loss = 0.1 * extremity_loss + 0.005 * parameters_regression_loss + 1000 * reconstruction_loss \
           + 0.02 * representation_diff_loss

    avg_loss += loss.item()
    loss = loss / batch_size
    loss.backward()
    # magnitude += torch.sum(torch.pow(agent.SoundEncoder.features2encoding.weight.grad, 2))
    # magnitude += torch.sum(torch.pow(agent.ParametersEncoder.encoding.weight.grad, 2))

    loss_parameters = loss_parameters + parameters_regression_loss.item()
    loss_reconstruction = loss_reconstruction + reconstruction_loss.item()
    loss_representation_diff = loss_representation_diff + representation_diff_loss.item()

    if count_batch == batch_size:
        opt.step()
        opt.zero_grad()
        count_batch = 0

    if count_test == 100:
        avg_loss /= 100
        loss_parameters /= 100
        loss_reconstruction /= 100
        loss_representation_diff /= 100
        magnitude /= 100
        # print(magnitude)

        wandb.log(
            {
                "parameter loss": loss_parameters,
                "reconstruction loss": loss_reconstruction,
                "representation diff loss": loss_representation_diff,
             }
        )
        count_test = 0
        loss_parameters = 0
        loss_reconstruction = 0
        loss_representation_diff = 0
        # magnitude = 0

wandb.finish()

torch.save(agent.SoundEncoder.state_dict(), fr'{file_dir}\SoundEncoder.pth')
torch.save(agent.ParametersEncoder.state_dict(), fr'{file_dir}\ParametersEncoder.pth')
