import Dynamic_synth.SynthModules as SM
import torch
import json
import matplotlib.pyplot as plt
from RL_Synth import Models, Environment, Algorithms, Utils
import torch.optim as optim
import os
import cProfile
import wandb

torch.cuda.memory._record_memory_history(enabled=True)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
synth_file = fr'{project_dir}/Dynamic_synth/Synth Configs/2OscillatorFreqWave.json'
training_config_file = fr'{project_dir}/RL_Synth/Training/Training_Configs/Training_Config_A2C.json'
environment_config_file = fr'{project_dir}/RL_Synth/Training/Environment_Configs/Environment_Config.json'
model_config_file = fr'{project_dir}/RL_Synth/Training/Model_Configs/CNN2D_small.json'
test_set_dir = fr'{project_dir}/RL_Synth/Training/Datasets/TestSet2Ofreq_wave.csv'
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
    lr_agent_representation = training_config["lr_agent_representation"]
    lr_critic_representation = training_config["lr_critic_representation"]
    lr_agent = training_config["lr_agent"]
    lr_critic = training_config["lr_critic"]
    exploratory = training_config["exploratory"]
    baseline = training_config["baseline"]
    algorithm = training_config["algorithm"]
    experiment_num = training_config["experiment_num"]
    training_config["experiment_num"] += 1

with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), device=device)

sound = my_synth(t).detach()
algorithm_class = Algorithms.algorithm2class[training_config['algorithm']]

env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=environment_config_file)

manager, critic, target_critic = Environment.initialize_agents(env=env,
                                                               training_config_dir=training_config_file,
                                                               model_config_dir=model_config_file,
                                                               is_single_agent=algorithm_class.is_single_agent)

agent_param_list = list()
# Building the parameter list for the optimizer.
# Stopping criteria has lower learning rate
level_to_model = Environment.DFS_agent_parameters(manager.top_agent)
# lr = lr_agent * 0.1 ** len(list(level_to_model.keys()))
n_levels = len(level_to_model.keys()) - 1
for level, agents_models in level_to_model.items():
    for model, model_name, sub_agent in agents_models:
        multiplier = max([0.1, 0.1 ** (n_levels - level)])
        # multiplier = 1
        # Higher learning rate for categorical choices
        if len(sub_agent.next_agents) == 0 and len(sub_agent.options) > 0:
            multiplier *= 10

        for sub_model_name, sub_model in model.named_children():
            if sub_model_name == 'termination':
                multiplier *= 0.1
                # multiplier *= 1

            param_dict = {'params': list(sub_model.parameters()), 'lr': lr_agent * multiplier}
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

if model_config["pretrained"]:
    sound_path_pretrained = fr"{project_dir}/Models/VF/cnn_vf_pretrain.pth"
    mlp_path_pretrained = fr"{project_dir}/Models/VF/mlp_vf_pretrain.pth"
    # sound_path_pretrained = fr"C:/Users/sgvfe/Desktop/Degree/2nd_degree/Thesis/Models/DeltaLab/enc_dec.ckpt"
    # parameters_path_pretrained = fr"C:/Users/sgvfe/Desktop/Degree/2nd_degree/Thesis/Models/SST/ParametersEncoder.pth"

    sound_model = Models.SimpleCNN([16, 16, 16], [3, 3, 3], representation_dim, env.n_frames, 128)
    state_dict = torch.load(sound_path_pretrained)
    sound_model.load_state_dict(state_dict)

    if model_config['are_shared_mlp']:
        mlp = Models.RLMLP(representation_dim, 1, env.curr_synth_parameters.shape[0], models_dim, n_layers)
        state_dict = torch.load(mlp_path_pretrained)
        mlp.load_state_dict(state_dict)
        manager.advanced_representation = mlp

    manager.SoundEncoder = sound_model
    critic.SoundEncoder = sound_model

manager.to(device=device)
critic.to(device=device)
target_critic.to(device=device)
critic.set_target_critic(target_critic)
critic.copy_weights_to_target()

if model_config["are_shared_backbone"]:
    param_list = critic_param_list + agent_param_list
    # optimizer = optim.Adam(params=param_list, lr=1e-6, betas=(0.9, 0.999), eps=1e-5)
    optimizer = optim.RMSprop(params=param_list, lr=1e-6, eps=1e-5)

    optimizer.add_param_group({'params': manager.SoundEncoder.parameters(), 'lr': lr_agent_representation})

    if model_config['are_shared_mlp']:
        optimizer.add_param_group({'params': manager.advanced_representation.parameters(), 'lr': lr_agent_representation})

    optimizer_agent = optimizer_critic = optimizer
    lr_scheduler_agent = lr_scheduler_critic = Utils.GradualWarmupScheduler(optimizer, total_steps=100)

else:
    optimizer_agent = optim.Adam(params=agent_param_list, lr=lr_agent, betas=(0.9, 0.99), eps=1e-5)
    # optimizer_agent = optim.RMSprop(params=agent_param_list, lr=1e-6, eps=1e-5)

    if training_config['is_ET']:
        optimizer_agent = Algorithms.AdamTrace(params=agent_param_list, beta1=training_config['lambda'],
                                               weight_decay=0.9)

    optimizer_critic = optim.Adam(params=critic_param_list, lr=lr_critic, betas=(0.3, 0.99))
    # optimizer_critic = optim.RMSprop(params=critic_param_list, lr=lr_critic)

    optimizer_agent.add_param_group({'params': manager.SoundEncoder.parameters(), 'lr': lr_agent_representation})
    optimizer_critic.add_param_group({'params': critic.SoundEncoder.parameters(), 'lr': lr_critic_representation})

    if model_config['are_shared_mlp']:
        optimizer_agent.add_param_group({'params': manager.advanced_representation.parameters(), 'lr': lr_agent_representation})
        optimizer_critic.add_param_group({'params': critic.advanced_representation.parameters(), 'lr': lr_critic_representation})

    lr_scheduler_agent = Utils.GradualWarmupScheduler(optimizer_agent, total_warmup_steps=5, total_zero_steps=0,
                                                      max_steps=50000, with_decay=False)
    lr_scheduler_critic = Utils.GradualWarmupScheduler(optimizer_critic, total_warmup_steps=5, total_zero_steps=0,
                                                      max_steps=50000, with_decay=False)

    optimizer_agent.zero_grad()
    optimizer_critic.zero_grad()

# breakpoint()
R = list()
run = None
logger = None
advantage_logger = None
T = 30

for T, n_steps in zip([30, 35, 40, 45], [15000, 15000, 15000, 15000]):
# for T, n_modules, n_parameters, n_steps in zip([30, 30, 30, 40, 50, 60],
#                                     [1, 1, 2, 2, 2, 2],
#                                     [1, 2, 1, 2, 2, 2],
#                                     [1000, 1000, 10000, 10000, 10000, 10000]):
    RL_optimizer = Algorithms.algorithm2class[algorithm](optimizer_agent=optimizer_agent,
                                                         optimizer_critic=optimizer_critic,
                                                         lr_scheduler_agent=lr_scheduler_agent,
                                                         lr_scheduler_critic=lr_scheduler_critic,
                                                         env=env,
                                                         agent=manager,
                                                         critic=critic,
                                                         are_shared_backbone=model_config['are_shared_backbone'],
                                                         is_pretrained=model_config['pretrained'],
                                                         n_steps=n_steps,
                                                         T=T,
                                                         n_modules=1,
                                                         n_parameters=1,
                                                         config=training_config,
                                                         test_set=test_set,
                                                         run=run,
                                                         logger=logger,
                                                         advantage_logger=advantage_logger)
    R = RL_optimizer.optimize()
    run = RL_optimizer.run
    logger = RL_optimizer.logger
    advantage_logger = RL_optimizer.advantage_logger
    R.append(R)
wandb.finish()
pr = cProfile.Profile()
pr.enable()
pr.disable()
pr.dump_stats('profile.pstat')

# for i in range(2, n_levels + 1):
#     with open(fr'Level{i}_Synth.json') as f:
#         synth_config2 = dict(json.load(f))
#         agent.level_up(synth_config2)
#
#     if training_config["baseline"] or algorithm == "ActorCritic":
#         critic.level_up(synth_config)
#     RL_optimizer = Algorithms.algorithm2class[algorithm](optimizer_agent=optimizer_agent,
#                                                          optimizer_critic=optimizer_critic,
#                                                          lr_scheduler_agent=lr_scheduler_agent,
#                                                          lr_scheduler_critic=lr_scheduler_critic,
#                                                          env=env,
#                                                          agent=agent,
#                                                          critic=critic,
#                                                          n_steps=n_steps[1],
#                                                          T=T[1],
#                                                          config=training_config)
#
#     R.append(RL_optimizer.optimize())

file_dir = fr"C:/Users/sgvfe/Desktop/Degree/2nd_degree/Thesis/Models/{algorithm}/{experiment_num}"
os.mkdir(file_dir)

agent_dir = fr"C:/Users/sgvfe/Desktop/Degree/2nd_degree/Thesis/Models/{algorithm}/{experiment_num}/Agent"
os.mkdir(agent_dir)
manager.save_models(agent_dir)

with open(fr'{file_dir}/config.json', 'w') as f:
    json.dump(training_config, f)

if training_config["baseline"] or algorithm == 'ActorCritic' or algorithm == 'OptionCritic':
    critic_dir = fr"C:/Users/sgvfe/Desktop/Degree/2nd_degree/Thesis/Models/{algorithm}/{experiment_num}/Critic"
    os.mkdir(critic_dir)
    critic.save_models(critic_dir)

with open(training_config_file, 'wt') as f:
    json.dump(training_config, f)
