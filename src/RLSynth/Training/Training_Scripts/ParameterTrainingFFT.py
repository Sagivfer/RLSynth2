import Dynamic_synth.SynthModules as SM
import torch
import json
import matplotlib.pyplot as plt
from RL_Synth import Models, Environment, Algorithms, Utils
import os
import wandb


synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreq.json'
training_config_file = r'../Training_Configs/Training_Config_A2C.json'
environment_config_file = r'../Environment_Configs/Environment_Config.json'
model_config_file = r'../Model_Configs/CNN2D_small.json'
test_set_dir = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\RL_Synth\Training\Datasets\TestSet10Step2Freq.csv'
test_set = Utils.create_test_set_list(test_set_dir)

with open(model_config_file) as f:
    model_config = dict(json.load(f))
    representation_dim = model_config["representation_dim"]
    models_dim = model_config["models_dim"]
    n_layers = model_config["n_layers"]

with open(training_config_file) as f:
    training_config = dict(json.load(f))
    device = training_config["device"]
    n_steps = training_config["n_steps"]
    T = training_config["episode_len"]
    lr_representation = training_config["lr_representation"]
    lr_agent = training_config["lr_agent"]
    lr_critic = training_config["lr_critic"]
    exploratory = training_config["exploratory"]
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

t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), device=device)
n_levels = 1

sound = my_synth(t).detach()
algorithm_class = Algorithms.algorithm2class[training_config['algorithm']]

env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=environment_config_file)

manager, critic, target_critic = Environment.initialize_agents(env=env,
                                                               training_config_dir=training_config_file,
                                                               model_config_dir=model_config_file,
                                                               is_single_agent=algorithm_class.is_single_agent)

parameter_regression = Environment.SynthRegression(env.synth2copy, representation_dim, models_dim, n_layers)
parameter_regression.to('cuda')

agent_param_list = list()
# Building the parameter list for the optimizer.
# Stopping criteria has lower learning rate
level_to_model = Environment.DFS_agent_parameters(manager.top_agent)
# lr = lr_agent * 0.1 ** len(list(level_to_model.keys()))
for level, agents_models in level_to_model.items():
    # lr = lr_agent * (0.5 ** (2 - level))
    lr = lr_agent
    for model, model_name, sub_agent in agents_models:
        for sub_model_name, sub_model in model.named_children():
            if sub_model_name == 'termination':
                param_dict = {'params': list(sub_model.parameters()), 'lr': lr}
            else:
                param_dict = {'params': list(sub_model.parameters()), 'lr': lr}
            agent_param_list.append(param_dict)
    # lr *= 10
manager.model_parameters = agent_param_list

critic_models, critic_model_names = critic.get_models(models=[], names=[])
critic_param_list = list()
for model, name in zip(critic_models, critic_model_names):
    if name != 'SoundEncoder' and name != 'advanced_representation':
        for p_name, param in model.named_parameters(recurse=True):
            param_dict = {'params': param, 'lr': lr_critic}
            critic_param_list.append(param_dict)
critic.model_parameters = critic_param_list

opt = torch.optim.Adam(agent_param_list)
opt.add_param_group({'params': manager.SoundEncoder.parameters(), 'lr': lr_representation})
opt.add_param_group({'params': manager.advanced_representation.parameters(), 'lr': lr_representation})
opt.add_param_group({'params': parameter_regression.parameters(), 'lr': lr_representation})

warmup_scheduler = Utils.GradualWarmupScheduler(opt, total_steps=1000)

manager.to(device=device)
critic.to(device=device)
target_critic.to(device=device)
critic.set_target_critic(target_critic)
critic.copy_weights_to_target()

count_batch = 0
count_test = 0
loss = 0
run = wandb.init(project='Synth_with_RL')
for epoch in range(100000):
    print(epoch)
    avg_loss = 0
    batch_n = 0
    env.randomize_synth(randomize_type=2)
    env.update_signal(update_type=2)
    sound_rep = manager.get_sound_representation(env.desired_signal)
    parameters, parameter_tensor = parameter_regression(sound_rep)

    sorted_oscillators = env.synth2copy.oscillators.copy()
    sorted_oscillators = sorted(sorted_oscillators, key=lambda o: o.freq.value)
    for i, oscillator in enumerate(env.synth2copy.oscillators):
        freq_target = oscillator.freq.value / oscillator.freq.max_val
        loss = loss + torch.pow(parameters[0][i][0][0] - freq_target, 2)

    if count_batch == batch_size:
        loss = loss / batch_size
        avg_loss += loss.item()
        # rep_size = rep_size / batch_size
        loss.backward()
        opt.step()
        opt.zero_grad()
        warmup_scheduler.step()
        wandb.log({"loss": loss}, step=epoch)
        loss = 0
        count_batch = 0

    if count_test == 10000:
        test_loss, test_freq_diff, test_rmse = Environment.test_model(test_set, 1, env, manager,
                                                                      parameter_regression=parameter_regression)
        wandb.log({'test_loss': test_loss,
                   'test_freq_diff': test_freq_diff,
                   'test_rmse': test_rmse},
                  step=epoch)
        count_test = 0

    count_batch += 1
    count_test += 1

wandb.finish()

torch.save(model.state_dict(), model_path)
