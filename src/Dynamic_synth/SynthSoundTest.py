import Dynamic_synth.SynthModules as SM
import torch
import json
from RL_Synth import Models, Environment, Algorithms, Utils
from matplotlib import pyplot as plt
import os
import cProfile


torch.cuda.memory._record_memory_history(enabled=True)

synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreq2Filter.json'
environment_config_dir = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\RL_Synth\Training\Environment_Configs\Environment_Config.json'

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

plt.plot(sound.cpu())
plt.show()

plt.imshow(env.desired_signal_mel_spec.cpu())
plt.show()

env.synth.play_sound()
env.synth.oscillators[1].phase.value = 0.2 * SM.TWO_PI
# env.synth.play_sound()
# env.synth.oscillators[1].phase.value = 0.4 * SM.TWO_PI
# env.synth.play_sound()
# env.synth.oscillators[1].phase.value = 0.6 * SM.TWO_PI
# env.synth.play_sound()
