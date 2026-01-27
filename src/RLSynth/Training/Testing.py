import src.Dynamic_synth.SynthModules as SM
import torch
import json
import matplotlib.pyplot as plt
from src.RLSynth import Models, Environment, Algorithms, Utils
import torch.optim as optim
import os
import cProfile

torch.cuda.memory._record_memory_history(enabled=True)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
synth_file = fr'{project_dir}/Dynamic_synth/Synth Configs/OscillatorAllFilterFreq.json'
environment_config_file = fr'{project_dir}/RLSynth/Training/Environment_Configs/Environment_Config.json'
model_config_file = fr'{project_dir}/RLSynth/Training/Model_Configs/CNN2D_small.json'
test_set_dir = fr'{project_dir}/RLSynth/Training/Datasets/TestSetOallFfreq.csv'
test_set = Utils.create_test_set_list(test_set_dir)

experiment_num = 1160
algorithm = 'OptionCritic'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_config_file = fr'{project_dir}/TrainedModels/{algorithm}/{experiment_num}/config.json'
model_config = json.load(open(training_config_file))

with open(model_config_file) as f:
    model_config = dict(json.load(f))
    representation_dim = model_config["representation_dim"]
    models_dim = model_config["models_dim"]
    n_layers = model_config["n_layers"]

with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device=device)

algorithm_sound_dir = fr"{project_dir}/Sounds/{algorithm}"
if not os.path.isdir(algorithm_sound_dir):
    os.mkdir(algorithm_sound_dir)

sound_dir = fr"{project_dir}/Sounds/{algorithm}/{experiment_num}"
if not os.path.isdir(sound_dir):
    os.mkdir(sound_dir)

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
manager.load_models(fr'{project_dir}/TrainedModels/{algorithm}/{experiment_num}/Agent')

manager.to(device=device)
critic.to(device=device)
target_critic.to(device=device)
critic.set_target_critic(target_critic)
critic.copy_weights_to_target()

pr = cProfile.Profile()
pr.enable()
Environment.test_model(test_set, 30, env, manager, play_sound=False, show_spec=False, save_sound=True,
                       sound_dir=sound_dir)
pr.disable()
pr.dump_stats('profile2.pstat')
