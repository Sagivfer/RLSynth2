import Dynamic_synth.SynthModules as SM
import torch
import json
import matplotlib.pyplot as plt
from RL_Synth import Models, Environment, Algorithms, Utils
import torch.optim as optim
import os
import cProfile

torch.cuda.memory._record_memory_history(enabled=True)

synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreq.json'
training_config_file = fr'Training_Configs/Training_Config_A2C.json'
environment_config_file = fr'Environment_Configs/Environment_Config.json'
model_config_file = r'Model_Configs/CNN2D_small.json'
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

with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), device=device)
n_levels = 1

sound = my_synth(t).detach()
# algorithm_class = Algorithms.algorithm2class[training_config['algorithm']]
algorithm_class = Algorithms.algorithm2class['PPOC']

env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=environment_config_file)

manager, critic, target_critic = Environment.initialize_agents(env=env,
                                                               training_config_dir=training_config_file,
                                                               model_config_dir=model_config_file,
                                                               is_single_agent=algorithm_class.is_single_agent)
env.tolerance = 0.0005
manager.load_models(fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\Models\OptionCritic\1141\Agent')

manager.to(device=device)
critic.to(device=device)
target_critic.to(device=device)
critic.set_target_critic(target_critic)
critic.copy_weights_to_target()

pr = cProfile.Profile()
pr.enable()
Environment.test_model(test_set, 20, env, manager, play_sound=False, show_spec=False)
pr.disable()
pr.dump_stats('profile2.pstat')
