import numpy as np
from RL_Synth.Utils import earth_mover_distance_averaged, earth_mover_distance
import RL_Synth.Environment as Environment
import RL_Synth.Models as Models
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import os

synth_file = fr'C:\Users\sgvfe\Desktop\Degree\2nd_degree\Thesis\ai_synth\src\Dynamic_synth\Synth Configs\2OscillatorFreqWave.json'

env = Environment.initialize_env(synth_config_dir=synth_file,
                                 environment_config_dir=r'Training_Configs/LossPlotConfig.json')


def animate(i):
    global env, im
    env.synth.filters[0].cutoff_freq = env.synth.filters[0].cutoff_freq + 10
    env.update_signal(update_type=1)
    mel_spec = env.current_signal_mel_spec.squeeze(0).cpu().numpy()
    im.set_array(mel_spec)
    return im,


resolution = 1000
fig = plt.figure()
synth_parameter2change1 = env.synth.oscillators[0].freq
synth_parameter2change2 = env.synth.oscillators[1].freq
synth_parameter2change3 = env.synth.oscillators[0].waveform
synth_parameter2change4 = env.synth.oscillators[1].waveform
synth2copy_parameter2change1 = env.synth2copy.oscillators[0].freq
synth2copy_parameter2change2 = env.synth2copy.oscillators[1].freq
synth2copy_parameter2change3 = env.synth2copy.oscillators[0].waveform
synth2copy_parameter2change4 = env.synth2copy.oscillators[1].waveform
for i in range(15):
    env.randomize_synth(randomize_type=3)
    # env.align_synth_parameters()
    # fig, axes = plt.subplots(1, 2)
    # mel_spec1 = env.current_signal_mel_spec.squeeze(0).cpu().numpy()
    # mel_spec2 = env.desired_signal_mel_spec.squeeze(0).cpu().numpy()
    # im1 = axes[0].imshow(mel_spec1, interpolation='none')
    # im2 = axes[1].imshow(mel_spec2, interpolation='none')
    # plt.show()
    freq1str = str(synth2copy_parameter2change1.value)
    freq2str = str(synth2copy_parameter2change2.value)
    dir_name = f'FreqWaveLoss/{freq1str[:6]}_{freq2str[:6]}_{synth2copy_parameter2change3.value}_{synth2copy_parameter2change4.value}'
    os.mkdir(dir_name)
    for option1 in synth_parameter2change3.options:
        synth_parameter2change3.value = option1
        for option2 in synth_parameter2change4.options:
            synth_parameter2change4.value = option2
            synth_parameter2change1.value = synth_parameter2change1.min_val
            synth_parameter2change2.value = synth_parameter2change2.min_val
            env.update_signal(update_type=3)
            losses = list()
            points = list()
            xs = list()
            ys = list()
            ax = plt.axes(projection='3d')
            while synth_parameter2change1.value < synth_parameter2change1.max_val:
                xs.append(synth_parameter2change1.value)
                ys.append(synth_parameter2change1.value)
                losses_point = list()
                synth_parameter2change1.value = synth_parameter2change1.value + synth_parameter2change1.max_val / resolution
                print(f"parameter1 = {synth_parameter2change1.value}")
                synth_parameter2change2.value = synth_parameter2change2.min_val
                while synth_parameter2change2.value < synth_parameter2change2.max_val:
                    print(f"parameter2 = {synth_parameter2change2.value}")
                    synth_parameter2change2.value = synth_parameter2change2.value + synth_parameter2change2.max_val / resolution
                    env.update_signal(update_type=1)
                    env.update_value()
                    loss = -env.curr_value
                    losses_point.append(loss)
                losses.append(losses_point)
            X = np.array(xs)
            Y = np.array(ys)
            Xv, Yv = np.meshgrid(X, Y)
            losses_array = np.array(losses)
            # ax.scatter(xs, ys, losses, s=1)
            surf = ax.plot_surface(Xv, Yv, losses_array, cmap=cm.coolwarm)
            ax.scatter(synth2copy_parameter2change1.value, synth2copy_parameter2change2.value, 0, s=15, color='r')
            ax.scatter(synth2copy_parameter2change2.value, synth2copy_parameter2change1.value, 0, s=15, color='r')
            plt.title(f'{option1}, {option2} vs {synth2copy_parameter2change3.value}, {synth2copy_parameter2change4.value}')
            plt.legend(loc='lower center')
            plt.colorbar(surf)
            plt.savefig(f'{dir_name}/loss{i}_{option1}_{option2}.png', dpi=300)
            ax.view_init(0, -90)
            plt.savefig(f'{dir_name}/loss{i}_{option1}_{option2}_side1.png', dpi=300)
            ax.view_init(0, 90)
            plt.savefig(f'{dir_name}/loss{i}_{option1}_{option2}_side2.png', dpi=300)
            # plt.show()
            plt.cla()
            plt.clf()
