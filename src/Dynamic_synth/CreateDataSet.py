import csv
import os
import Dynamic_synth.SynthModules as SM
import torch
import json
from RL_Synth import Environment


torch.cuda.memory._record_memory_history(enabled=True)

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
synth_file = fr'{project_dir}/Dynamic_synth/Synth Configs/OscillatorAllFilterFreq.json'
environment_config_dir = fr'{project_dir}/RL_Synth/Training/Environment_Configs/Environment_Config.json'


with open(synth_file) as f:
    synth_config = dict(json.load(f))
    sample_rate = synth_config["sample_rate"]
    signal_duration = synth_config["signal_duration"]
    my_synth = SM.build_synth(synth_config, device='cuda')

t = torch.linspace(0, signal_duration, steps=int(sample_rate * signal_duration), device='cuda')
n_levels = 1

sound = my_synth(t).detach()
env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=environment_config_dir)

n_configs = 10000
for n_step in range(10, 11):
    param_list = list()
    for i in range(n_configs):
        env.randomize_synth(randomize_type=3)
        parameters2copy = env.synth2copy.get_synth_parameters()
        parameters = env.synth.get_synth_parameters()
        param_list.append(parameters2copy)
        param_list.append(parameters)

    with open(f'../RL_Synth/Training/Datasets/TestSetOallFfreq.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in param_list:
            writer.writerow(row)

