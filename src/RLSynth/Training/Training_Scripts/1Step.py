import csv

import matplotlib.pyplot as plt
import Dynamic_synth.SynthModules as SM
import torch
import json
from RL_Synth import Models, Environment, Algorithms, Utils
from utils import metrics
import os
import wandb
import numpy as np
import cProfile


def get_action(env: Environment.Environment):
    actions = list()
    for i, oscillator in enumerate(env.synth.oscillators):
        closest_oscillator = min(env.synth2copy.oscillators, key=lambda o: abs(o.freq.value - oscillator.freq.value))
        target_action = closest_oscillator.freq.value - oscillator.freq.value
        target_action = min([target_action, 375])
        target_action = max([target_action, -375])
        actions.append(target_action)

    return actions


synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreq.json'
training_config_file = fr'../Training_Configs/Training_Config_1Step.json'
environment_config_file = fr'../Environment_Configs/Environment_Config.json'

test_set_dir = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\RL_Synth\Training\Datasets\TestSet10Step2Freq.csv'
test_set = Utils.create_test_set_list(test_set_dir)

with open(training_config_file) as f:
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

with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=environment_config_file)

agent, critic, target_critic = Environment.initialize_agents(env=env,
                                                             training_config_dir=training_config_file,
                                                             is_single_agent=True)

agent.to('cuda')
critic.to('cuda')

agent_param_list = list()
# Building the parameter list for the optimizer.
# Stopping criteria has lower learning rate
level_to_model = Environment.DFS_agent_parameters(agent.top_agent)
# lr = lr_agent * 0.1 ** len(list(level_to_model.keys()))
for level, agents_models in level_to_model.items():
    lr = lr_agent
    for model, model_name, sub_agent in agents_models:
        for sub_model_name, sub_model in model.named_children():
            if sub_model_name == 'termination':
                param_dict = {'params': list(sub_model.parameters()), 'lr': lr}
            else:
                param_dict = {'params': list(sub_model.parameters()), 'lr': lr}
            agent_param_list.append(param_dict)
    # lr *= 10
agent.model_parameters = agent_param_list

opt = torch.optim.Adam(agent_param_list)
opt.add_param_group({'params': agent.SoundEncoder.parameters(), 'lr': lr_representation})
opt.add_param_group({'params': agent.advanced_representation.parameters(), 'lr': lr_representation})

warmup_scheduler = Utils.GradualWarmupScheduler(opt, total_steps=100, total_zero_steps=50)
wandb.init(project='Synth_with_RL')

count_batch = 0
count_test = 0
loss = 0
rep_size = 0
magnitude = 0


pr = cProfile.Profile()
pr.enable()
for t in range(2500000):
    print(t)
    count_batch += 1
    count_test += 1
    env.generate_starting_point(n_step=1)
    env_value = env.get_value()

    state = env.get_state()

    state_representation, _ = agent.get_state_representation(state)
    target_actions = get_action(env)

    policy, _, logits = agent.get_policy_from_state_representation(state_representation)
    for sub_policy, action in zip(policy, target_actions):
        loss = loss + torch.pow((sub_policy[0] - action) / 375, 2)

    rep_size = rep_size + this_rep_size

    if count_batch == batch_size:
        loss = loss / batch_size
        rep_size = rep_size / batch_size
        loss.backward()
        opt.step()
        opt.zero_grad()
        warmup_scheduler.step()
        # if t > 1000:
        wandb.log({"loss": loss,
                   "rep_size": rep_size
                   }, step=t)
        loss = 0
        count_batch = 0

    if count_test == 10000:
        test_loss, freq_diff = Environment.test_model(test_set, 1, env, agent)
        wandb.log({"test_loss": test_loss,
                   "test_freq_diff": freq_diff,
                   }, step=t)
        count_test = 0

pr.disable()
pr.dump_stats('profile.pstat')
wandb.finish()

out_file_dir = fr"C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\1Step"

try:
    os.mkdir(out_file_dir)
except:
    pass

with open(fr'{out_file_dir}\config.json', 'w') as f:
    json.dump(training_config, f)

with open(training_config_file, 'wt') as f:
    json.dump(training_config, f)

cnn_model_path = fr'{out_file_dir}\cnn_vf_pretrain.pth'
mlp_model_path = fr'{out_file_dir}\mlp_vf_pretrain.pth'
torch.save(agent.SoundEncoder.state_dict(), cnn_model_path)
torch.save(agent.advanced_representation.state_dict(), mlp_model_path)
