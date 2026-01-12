import matplotlib.pyplot as plt

import Dynamic_synth.SynthModules as SM
import torch
import json
from src.RLSynth import Models, Environment, Algorithms, Utils
import os
import wandb
import numpy as np


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

with open(fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Full_Synth.json') as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

env, agent, critic = Environment.initialize_env(config_dir=r'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Full_Synth.json',
                                                training_config_dir=r'../Training Configs/Training_Config.json')
file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\SRL"
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
    for model, model_name, sub_agent in agents_models:
        for sub_model_name, sub_model in model.named_children():
            param_dict = {'params': list(sub_model.parameters()), 'lr': 1e-3}
            agent_param_list.append(param_dict)
agent.model_parameters = agent_param_list

critic_models, critic_model_names = critic.get_models(models=[], names=[])
critic_param_list = list()
for model, name in zip(critic_models, critic_model_names):
    if name != 'SoundEncoder':
        for p_name, param in model.named_parameters(recurse=True):
            param_dict = {'params': param, 'lr': lr_critic}
            critic_param_list.append(param_dict)
critic.model_parameters = critic_param_list

opt = torch.optim.RMSprop(agent_param_list + critic_param_list, weight_decay=0.01, lr=5e-6)
opt.add_param_group({'params': agent.SoundEncoder.parameters()})
wandb.init(project='Synth_with_RL')
count_switch_agents = 0
count_randomize_synth = 0
count_batch = 0

avg_loss = 0
loss_action = 0
loss_state = 0
loss_value = 0
loss_parameters = 0
loss = 0
magnitude = 0
env.randomize_synth(randomize_type=1)
current_agent, agent_indices = get_random_agents(agent)

update_diversity = False
prev_for_diversity = None

parameters = ["state regression loss", "action regression loss", "value estimation loss", "parameter loss",
              "freq", "waveform",
              "cutoff_freq", "resonance", "filter_type",
              "attack", "decay", "sustain", "release", "sustain_time"]

count_parameters = {parameter: 0 for parameter in parameters}
what2log = {parameter: 0 for parameter in parameters}

fig, ax = plt.subplots()
for t in range(1000000):
    print(t)

    if count_randomize_synth == 10:
        avg_loss = 0
        count_parameters = {parameter: 0 for parameter in parameters}
        what2log = {parameter: 0 for parameter in parameters}
        loss = 0
        magnitude = 0

        env.randomize_synth(randomize_type=1)
        current_agent, agent_indices = Utils.get_random_agents(agent)

        update_diversity = False
        prev_for_diversity = None

    if count_switch_agents == 1:
        current_agent, agent_indices = Utils.get_random_agents(agent)
        count_switch_agents = 0

    count_switch_agents += 1
    count_randomize_synth += 1
    count_batch += 1

    action_temp = list(current_agent.get_random_action())
    action = agent_indices + action_temp

    prev_signal_representation = agent.get_sound_features(env.current_signal)
    prev_synth_state = env.curr_synth_parameters.to(torch.float32)
    prev_mel_spec = env.current_signal_mel_spec
    p = np.random.uniform()
    if p < 0.1:
        prev_for_diversity = prev_signal_representation.detach()
        update_diversity = True

    env.step(action=action)

    current_signal_representation = agent.get_sound_features(env.current_signal)
    current_mel_spec = env.current_signal_mel_spec
    state_prediction, action_prediction = current_agent.get_SRL_predictions(action, prev_synth_state,
                                                                            prev_signal_representation,
                                                                            current_signal_representation)

    state_regression_loss = torch.mean(torch.pow(state_prediction - current_signal_representation.detach(), 2))
    what2log["state regression loss"] += state_regression_loss.item()

    if current_agent.selection is not None:
        ce_loss = torch.nn.CrossEntropyLoss()
        action_regression_loss = 0.5 * ce_loss(action_prediction, action[3].detach())
        # print(current_agent.module_name, action[3], action_prediction, action_regression_loss)
    else:
        action_regression_loss = torch.pow(action_prediction - action[2].detach() / current_agent.regression.max_step_val, 2)

    what2log["action regression loss"] += action_regression_loss.item()
    what2log[current_agent.module_name] += action_regression_loss
    count_parameters[current_agent.module_name] += 1

    diversity_loss = torch.tensor(0)
    if update_diversity:
        update_diversity = False
        diversity = torch.mean(torch.pow(current_signal_representation - prev_for_diversity, 2))
        diversity_loss = torch.exp(-1 * diversity)

    parameters_regression_loss = agent.get_parameters_synth_loss(current_signal_representation)
    what2log["parameter loss"] += parameters_regression_loss.item()

    loss = 0.0033 * state_regression_loss + 0.002 * action_regression_loss \
           + 500 * diversity_loss + 0.00003 * parameters_regression_loss

    loss.backward()
    # if current_agent.action_regression is not None and current_agent.action_regression.action.weight.grad is not None:
    #     print(f"{current_agent.module_name} Regression")
    #     print(torch.sum(torch.pow(current_agent.action_regression.action.weight.grad, 2)))
    #     print(torch.sum(torch.pow(current_agent.action_regression.action.weight, 2)))

    # if current_agent.state_regression is not None and current_agent.state_regression.state.weight.grad is not None:
    #     print(f"{current_agent.module_name} Selection")
    #     print(torch.sum(torch.pow(current_agent.state_regression.state.weight.grad, 2)))
    #     print(torch.sum(torch.pow(current_agent.state_regression.state.weight, 2)))
    # print(torch.sum(torch.pow(agent.SoundEncoder.features2encoding.weight.grad, 2)))
    # print(torch.sum(torch.pow(agent.SoundEncoder.features2encoding.weight, 2)))
    # magnitude += torch.sum(torch.pow(agent.SoundEncoder.features2encoding.weight.grad, 2))

    loss_action = loss_action + action_regression_loss.item()
    loss_state = loss_state + state_regression_loss.item()
    loss_parameters = loss_parameters + parameters_regression_loss.item()

    if count_batch == batch_size:
        opt.step()
        opt.zero_grad()
        count_batch = 0

    if count_randomize_synth == 100:
        count_parameters["state regression loss"] = 100
        count_parameters["action regression loss"] = 100
        count_parameters["value estimation loss"] = 100
        count_parameters["parameter loss"] = 100

        for parameter in parameters:
            if count_parameters[parameter] > 0:
                what2log[parameter] /= count_parameters[parameter]

        if t > 1000:
            wandb.log(what2log)
        count_randomize_synth = 0

wandb.finish()

model_path = fr'{file_dir}\synth_no_lfo5.pth'
torch.save(agent.SoundEncoder.state_dict(), model_path)
