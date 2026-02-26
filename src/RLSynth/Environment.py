import torch
import torch.nn as nn
import numpy as np
from src.Dynamic_synth import SynthModules as SM
from src.RLSynth import Models
from src.RLSynth import Utils
import json
import os
import wandb
from abc import ABC, abstractmethod
from random import shuffle, sample
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# from RL_Synth.Training.Training_Scripts.ParameterTrainingFFT import oscillator
# from RL_Synth.Training.Training_Scripts.SST import current_agent

idx2waveform = {0: 'sine', 1: 'square', 2: 'triangle', 3: 'sawtooth'}
waveform2idx = {'sine': 0, 'square': 1, 'triangle': 2, 'sawtooth': 3}

idx2filter = {0: 'lowpass', 1: 'highpass', 2: 'bandpass'}
filter2idx = {'lowpass': 0, 'highpass': 1, 'bandpass': 2}

config_module2model = {'O': 'Oscillator',
                       'F': 'Filter',
                       'ADSR': 'ADSR'}


eps = 0.0001


class Environment(object):
    """

    """
    def __init__(self, synth_config, hierarchy_type, is_dynamic_synth: bool, randomize_time=6, device='cpu',
                 reward_type=['similarity'], reward_weight=[1], max_state_val=1, scale_factor=1.0, reward_offset=0,
                 starting_tolerance=0.05, ending_tolerance=0.01, edge_penalty=0, action_scale='linear'):
        self.device = device

        self.set_synth_related_attributes(synth_config)

        self.mel_spec = Utils.mel_spectrogram_transform.to(self.device)
        self.n_frames = int(self.synth.sample_rate * self.synth.signal_duration / self.mel_spec.hop_length + 1)

        self.desired_signal = self.synth2copy(self.synth.t).to(self.device)
        self.current_signal = self.synth(self.synth.t).to(self.device)
        self.desired_signal_fft = torch.fft.rfft(self.desired_signal, norm='forward').abs()

        self.desired_signal_mel_spec = self.get_mel_spec(self.desired_signal)
        self.current_signal_mel_spec = self.get_mel_spec(self.current_signal)

        self.reward_type = reward_type
        self.reward_weight = reward_weight
        self.max_scaled_val = max_state_val / scale_factor
        self.scale_factor = scale_factor
        self.update_value()

        self.mel_spec = Utils.mel_spectrogram_transform.to(self.device)

        self.previous_action = None

        self.is_dynamic_synth = is_dynamic_synth
        self.hierarchy_type = hierarchy_type

        self.curr_synth_parameters = self.get_synth_parameters()

        self.randomize_timer = 0
        self.randomize_time = randomize_time

        self.reward_offset = reward_offset
        self.edge_penalty = edge_penalty
        self.action_scale = action_scale
        self.tolerance = starting_tolerance
        self.ending_tolerance = ending_tolerance

    def set_synth_related_attributes(self, synth_config):
        self.config = synth_config
        self.synth = SM.build_synth(synth_config, is_init=False, device=self.device)
        self.synth2copy = SM.build_synth(synth_config, device=self.device)
        self.synth_modules = self.synth.modules
        self.first_modules = self.create_first_module_options()
        self.second_modules = self.create_second_module_options()
        self.module2index = dict((module, i) for i, module in enumerate(self.synth_modules))
        self.synth_graph = self.calc_synth_graph()

    def get_synth_parameters(self) -> torch.Tensor:
        """
        Returns a tensor representing the synth parameters and the synth connections graph.
        """

        parameter_list = list()
        for module in self.synth.available_modules:
            parameter_list = parameter_list + module.get_module_vector()

        synth_state = np.array(parameter_list)

        if self.is_dynamic_synth:
            synth_graph = self.synth_graph.flatten()
            state = np.concatenate((synth_state, synth_graph))
        else:
            state = synth_state

        return torch.tensor(state, device=self.device).to(torch.float32)

    def create_first_module_options(self) -> list:
        """
        Create the list containing the modules that are optional as first module to be connected as input
        """

        module_list = list()
        for module in self.synth.editable_modules:
            module_list.append(module)
        return module_list

    def create_second_module_options(self) -> list:
        """
        Create the list containing the modules that are optional as second module to be connected as output
        """

        module_list = list()
        for module in list(vars(self.synth).keys()):
            if type(module) in self.synth.available_module_types:
                module_list.append(module)
        return module_list

    def calc_synth_graph(self) -> np.ndarray:
        """
            Creates a graph out of the synth modules.
            If module1's output is module2, then in the graph their would be 1 from 1 to 2, otherwise 0.
        """

        n_modules = len(self.synth_modules)
        synth_graph = np.zeros([n_modules, n_modules])
        for module in self.synth_modules:
            next_module = module.output_module
            if next_module is not None:
                synth_graph[self.module2index[module], self.module2index[next_module]] = 1
        return synth_graph

    def step(self, actions, inference_mode=False) -> (torch.float32, list):
        """
        Takes a series of actions to be take simultaneously
        Returns the reward and the next state from the actions
        """

        n_edges = 0
        previous_value = self.curr_value
        for action in actions:
            if self.hierarchy_type == 'module':
                changed_module = self.synth.editable_modules[action[0]]
                parameter = changed_module.editable_parameters[action[1]]
            else:
                changed_module = self.synth.editable_modules[action[1]]
                parameter = changed_module.editable_parameters[action[0]]
            parameter_object = changed_module.__getattribute__(parameter)
            parameter_diff = action[-3] if len(parameter_object.options) == 0 else parameter_object.options[action[-3].item()]
            parameter_value = parameter_object.value
            if len(parameter_object.options) != 0:
                parameter_object.value = parameter_diff
            else:
                if self.action_scale == 'linear':
                    parameter_object.value = parameter_object.value + parameter_diff.item()
                elif self.action_scale == 'log':
                    ratio = torch.exp(parameter_diff)
                    parameter_object.value = parameter_object.value * ratio.item()

            new_parameter_value = changed_module.__getattribute__(parameter).value
            # if new_parameter_value == parameter_value and len(parameter_object.options) == 0:
            if new_parameter_value == parameter_value:
                n_edges += 1

        self.update_signal(update_type=1)
        self.curr_synth_parameters = self.get_synth_parameters()
        self.update_value()

        edge_penalty = n_edges * self.edge_penalty
        reward = self.curr_value - previous_value + self.reward_offset + edge_penalty
        next_state = self.get_state()
        return reward, next_state

    def revert_action(self):
        """
        Reverts an action, not used currently
        """

        global idx2waveform, eps

        action = self.previous_action
        changed_module = self.synth.available_modules[action[0]]
        parameter = changed_module.available_parameters[action[1]]
        parameter_diff = action[2]

        if isinstance(parameter_diff, str):
            changed_module.__setattr__(parameter, parameter_diff)
        else:
            parameter_value = changed_module.__getattribute__(parameter)
            if self.action_scale == 'linear':
                changed_module.__setattr__(parameter, parameter_value - parameter_diff.item())
            elif self.action_scale == 'log':
                ratio = torch.exp(parameter_diff)
                changed_module.__setattr__(parameter, parameter_value / ratio.item())

        # if self.is_dynamic_synth and self.selected_modules[1] is not None:
        #     self.synth.connect_modules(self.selected_modules[0], self.selected_modules[1])

        self.update_signal(update_type=1)

        self.curr_synth_parameters = self.get_synth_parameters()

    def get_value(self, is_plot=False) -> torch.float32:
        """
        Returns the distance of a state based on the current signal and the desired signal.
        This method handles weighted average of multiple rewards.
        """

        distance = 0
        for reward_type, reward_weight in zip(self.reward_type, self.reward_weight):
            if reward_type == 'similarity':
                distance += reward_weight * torch.dot(self.current_signal, self.desired_signal)

            elif reward_type == 'earth_mover':
                distance += -1 * reward_weight * (Utils.magnitude_penalized_emd(
                    self.desired_signal_mel_spec.squeeze(0),
                    self.current_signal_mel_spec.squeeze(0))).detach().item()

            elif reward_type == 'diff_similarity':
                padded_left_current = torch.constant_pad_nd(self.current_signal, (1, 0))
                diff_current = padded_left_current[:-1] - self.current_signal
                padded_left_desired = torch.constant_pad_nd(self.desired_signal, (1, 0))
                diff_desired = padded_left_desired[:-1] - self.desired_signal
                distance += reward_weight * torch.dot(diff_current, diff_desired) /\
                            (torch.linalg.norm(diff_current) * torch.linalg.norm(diff_desired) + 0.001)

            elif reward_type == 'peak_diff':
                sum_current = Utils.get_sum_peaks(np.array(self.current_signal.cpu()))
                sum_desired = Utils.get_sum_peaks(np.array(self.desired_signal.cpu()))
                distance_peak = -1 * np.abs(sum_desired - sum_current) / (sum_current + sum_desired)
                distance += reward_weight * distance_peak

            elif reward_type == 'parameter_loss':
                for m1, m2 in zip(self.synth2copy.modules, self.synth.modules):
                    for parameter in m1.editable_parameters:
                        parameter_object1 = m1.__getattribute__(parameter)
                        parameter_object2 = m2.__getattribute__(parameter)
                        if len(parameter_object1.options) != 0:
                            if parameter_object1.value_index != parameter_object2.value_index:
                                distance += -1 * reward_weight * 1
                        else:
                            distance += -1 * reward_weight * \
                                    abs(parameter_object1.value - parameter_object2.value) / parameter_object1.max_val

        return -1 * distance / self.scale_factor

    def get_state(self) -> list:
        return [self.desired_signal, self.current_signal, self.curr_synth_parameters]

    def update_value(self, is_plot=False):
        self.curr_value = self.get_value(is_plot=is_plot)

    def randomize_synth(self, randomize_type=1):
        """
        Randomizes the synthesizers parameters.
        randomize_type:
            1 - randomizes the synth the agent manipulates
            2 - randomizes the synth the agent needs to copy
            3 - randomizes both
        """
        if randomize_type == 1 or randomize_type == 3:
            self.synth.randomize_parameters()
            if self.is_dynamic_synth:
                self.synth.randomize_flow()

        if randomize_type == 2 or randomize_type == 3:
            self.synth2copy.randomize_parameters()
            if self.is_dynamic_synth:
                self.synth2copy.randomize_flow()

    def align_synth_parameters(self, is_shuffle=False):
        """
        Aligns the parameters of the synths that are available yet not editable so they.
        Let's say we want to change the wave shape but just for generalization, we would set the desired and the current
        to be the same.
        """
        # initializing the values of the non-editable parameters as the same, and then aligning them
        for module, module2copy in zip(self.synth.modules, self.synth2copy.modules):
            for parameter in module.available_parameters:
                # If the parameter is not editable, set it as the synth2copy's
                parameter_object = getattr(module, parameter)
                parameter2copy_object = getattr(module2copy, parameter)
                parameter_object.value = parameter2copy_object.value

        # Shuffling the oscillators since there is a degree of freedom there
        if is_shuffle:
            parameter_lists = dict()
            bundles = [
                tuple(getattr(oscillator, parameter) for parameter in self.synth.oscillators[0].available_parameters)
                for oscillator in self.synth.oscillators
            ]

            shuffle(bundles)

            for oscillator, bundle in zip(self.synth.oscillators, bundles):
                for parameter_name, new_parameter_object in zip(self.synth.oscillators[0].available_parameters, bundle):
                    parameter_object = getattr(oscillator, parameter_name)
                    parameter_object.value = new_parameter_object.value

            # for parameter in :
            #     current_parameters = [getattr(oscillator, parameter) for oscillator in self.synth.oscillators]
            #     parameter_lists[parameter] = current_parameters
            #
            #     shuffled_oscillators = self.synth.oscillators[:]
            #     shuffle(shuffled_oscillators)
            #     for oscillator, shuffled_oscillator in zip(self.synth.oscillators, shuffled_oscillators):
            #         oscillator.freq.value = shuffled_oscillator.freq.value
            #         oscillator.waveform.value = shuffled_oscillator.waveform.value

    def update_signal(self, update_type=1):
        """
        Randomizes the synthesizers parameters.
        randomize_type:
            1 - updates the signal of the synth the agent manipulates
            2 - updates the signal of the synth the agent needs to copy
            3 - updates both signals
        """

        with torch.no_grad():
            if update_type == 1 or update_type == 3:
                self.current_signal = self.synth(self.synth.t)
                self.current_signal_mel_spec = self.get_mel_spec(self.current_signal)

            if update_type == 2 or update_type == 3:
                self.desired_signal = self.synth2copy(self.synth.t)
                self.desired_signal_mel_spec = self.get_mel_spec(self.desired_signal)

            # self.curr_synth_parameters = self.get_synth_parameters()

    def get_mel_spec(self, signal) -> torch.Tensor:
        signal_mel_spec = self.mel_spec(signal) / 150
        # signal_mel_spec = torch.clamp(signal_mel_spec, 0, 100)
        return signal_mel_spec

    def get_episode_stopping_condition(self, action_taken: bool = True) -> bool:
        terminate = False
        if self.curr_value > -1 * self.tolerance * self.max_scaled_val and action_taken:
            terminate = True
        return terminate

    def init_synth_parameter(self, initialization_type, parameter_object, parameter2copy_object, n_step=1):
        """
        Randomizes the synthesizers parameters.
        initialization_type:
            1 - sets the parameters of the current synth parameter close to the desired one (with random noise)
            2 - sets the parameters of the current synth parameter close to their minimum value
            3 - sets the parameters of the current synth parameter close to their maximum value
            4 - sets the parameters of the current synth parameter n-step away from the desired parameter
            99 - default exit
        """
        if initialization_type == 99:
            parameter_object.randomize_parameter()
            return

        noise = 0.1 * np.random.uniform(-1 * parameter_object.max_val, parameter_object.max_val)
        if initialization_type == 1:
            if len(parameter_object.options) == 0:
                parameter_object.value = parameter2copy_object.value + noise

        elif initialization_type == 2:
            if len(parameter_object.options) == 0:
                parameter_object.value = np.random.uniform(parameter_object.min_val,
                                                           parameter_object.min_val + 0.1 * parameter_object.max_val)

        elif initialization_type == 3:
            if len(parameter_object.options) == 0:
                parameter_object.value = np.random.uniform(parameter_object.max_val - 0.1 * parameter_object.max_val,
                                                           parameter_object.max_val)

        elif initialization_type == 4:
            noise = 0.1 * n_step * np.random.uniform(-1 * parameter_object.max_val, parameter_object.max_val)
            if len(parameter_object.options) == 0:
                parameter_object.value = parameter2copy_object.value + noise

        elif initialization_type == 5:
            if len(parameter2copy_object.options) == 0:
                parameter2copy_object.value = np.random.uniform(parameter2copy_object.min_val,
                                                                parameter2copy_object.min_val + 0.1 * parameter2copy_object.max_val)

        elif initialization_type == 6:
            if len(parameter2copy_object.options) == 0:
                parameter2copy_object.value = np.random.uniform(parameter2copy_object.max_val - 0.1 * parameter2copy_object.max_val,
                                                                parameter2copy_object.max_val)


    def generate_starting_point(self, is_hard_search=False):
        """
        Generates a starting point using init_synth_parameters with random generated initialization_type.
        This method also updates the signal
        """

        self.randomize_synth(randomize_type=2)  # Randomize the synth2copy
        # Align the parameters of the synths
        if not is_hard_search:
            self.align_synth_parameters(is_shuffle=True)

        # Giving 5% chance for each number below the number of editable modules.
        # When we have 3 modules as an example, we have 5% for 1 and 5% for 2 and 90% for 3.
        p_modules = np.random.random()
        n_modules_local = min([int(p_modules / 0.05) + 1, len(self.synth.editable_modules)])
        # n_modules_local = 2

        # Choosing just some of the modules to change to teach the model how to choose what module to change.
        chosen_module_indices = sample(range(len(self.synth.editable_modules)), n_modules_local)
        modules = [self.synth.editable_modules[i] for i in chosen_module_indices]
        modules2copy = [self.synth2copy.editable_modules[i] for i in chosen_module_indices]

        parameters_indices = list()
        for module in modules:
            p_parameters = np.random.random()
            n_parameters_local = min([int(p_parameters / 0.05) + 1, len(self.synth.editable_modules)])
            # n_parameters_local = 2

            # Choosing just some of the parameters to change to teach the model how to choose what parameter to change.
            if n_parameters_local > len(module.editable_parameters):
                chosen_parameter_indices = list(range(len(module.editable_parameters)))
            else:
                chosen_parameter_indices = sample(range(len(module.editable_parameters)), n_parameters_local)
            parameters_indices.append(chosen_parameter_indices)

        # Sampling the chosen parameters
        for module, module2copy, chosen_parameter_indices in zip(modules, modules2copy, parameters_indices):
            parameters = [module.editable_parameters[i] for i in chosen_parameter_indices]
            for parameter in parameters:
                parameter_object = getattr(module, parameter)
                parameter2copy_object = getattr(module2copy, parameter)

                p = np.random.random()
                initialization_type = 99

                # Initial guess of the point
                if p < 0:
                    initialization_type = 1

                # Closer to minimum values
                elif 0.1 <= p < 0.15:
                    initialization_type = 2

                # Closer to maximum values
                elif 0.15 <= p < 0.2:
                    initialization_type = 3

                # Target closer to minimum values
                elif 0.2 <= p < 0.3:
                    initialization_type = 5

                # Target closer to maximum values
                elif 0.3 <= p < 0.4:
                    initialization_type = 6

                self.init_synth_parameter(initialization_type, parameter_object, parameter2copy_object)

        self.update_signal(3)

    def decrease_tolerance(self):
        self.tolerance *= 0.95
        self.tolerance = max([self.tolerance, self.ending_tolerance])

    def save_sound(self, save_type):
        """
        Saves the sound generated by the synthesizer.
        save_type:
            1 - Saves the sound generated by the synthesizer the agent manipulates
            2 - Saves the sound generated by the synthesizer the agent needs to copy
            3 - Saves the sound generated by the synthesizer both
        """
        if save_type == 1 or save_type == 3:
            self.synth2copy.save_sound()

        if save_type == 2 or save_type == 3:
            self.synth.save_sound()


class Manager:
    def __init__(self, environment: Environment, representation_dim: int, models_dim: int, n_layers, device='cpu',
                 backbone_type='FusionModel', are_shared_mlp=False, activation='Tanh', is_exploratory=False):
        self.representation_dim = representation_dim
        self.models_dim = models_dim
        self.n_layers = n_layers
        self.activation = activation

        self.device = device
        self.environment = environment

        self.current_signal_features = None
        self.desired_signal_features = None

        self.backbone_type = backbone_type
        self.are_shared_mlp = are_shared_mlp
        self.need_mel_spec = False
        self.build_backbone()
        self.module_name = 'Manager'

    def build_backbone(self):
        if self.backbone_type == 'FusionModel':
            self.SoundEncoder = Models.FusionModel([24, 48, 64],
                                                   [3, 3, 3],
                                                   [24, 48, 64],
                                                   [3, 3, 3],
                                                   self.representation_dim,
                                                   self.models_dim, 3)
            self.need_mel_spec = True

        elif self.backbone_type == 'ResNet':
            self.SoundEncoder = Models.BackBone([2, 2, 2, 2],
                                                self.representation_dim,
                                                sample_rate=44100).to(torch.float32)
            self.need_mel_spec = True

        elif self.backbone_type == 'CNN2D':
            n_filters = [4, 4, 4]
            filter_sizes = [3, 3, 3]
            self.SoundEncoder = Models.SimpleCNN(n_filters,
                                                 filter_sizes,
                                                 self.representation_dim,
                                                 self.environment.n_frames,
                                                 n_sounds=1).to(torch.float32)
            self.need_mel_spec = True

        elif self.backbone_type == 'CNN1D':
            self.SoundEncoder = Models.SimpleCNN1D([32, 64, 96],
                                                   [9, 9, 9],
                                                   self.representation_dim).to(torch.float32)

        elif self.backbone_type == 'Attention':
            self.SoundEncoder = Models.AttentionModel(n_mels=self.environment.mel_spec.n_mels,
                                                      n_frames=self.environment.n_frames,
                                                      model_dim=self.models_dim,
                                                      representation_dim=self.representation_dim)
            self.need_mel_spec = True

        if self.are_shared_mlp:
            # self.advanced_representation = M.RLMLP(self.representation_dim, 2, self.environment.curr_synth_parameters.shape[0],
            #                                        self.models_dim, self.n_layers)
            self.advanced_representation = Models.RLMLP(representation_dim=self.representation_dim,
                                                        n_representations=1,
                                                        context_dim=self.environment.curr_synth_parameters.shape[0],
                                                        model_dim=self.models_dim,
                                                        # n_layers=self.n_layers,
                                                        n_layers=2,
                                                        activation='Tanh',
                                                        is_residual=True,
                                                        is_deep_wide=True)

    def get_sound_representation(self, signal1=None, signal2=None, mel_spec=None, mel_spec2=None) -> torch.Tensor:
        """
        Returns the representation that comes out of the backbone model
        if one of the parameters is not supplied, the current one in the environment is taken.
        """

        signal1_local = signal1 if signal1 is not None else self.environment.current_signal
        signal2_local = signal2 if signal2 is not None else self.environment.desired_signal

        if self.need_mel_spec:
            if mel_spec is None:
                mel_spec = self.environment.get_mel_spec(signal1_local)

            if signal2 is not None:
                if mel_spec2 is None:
                    mel_spec2 = self.environment.get_mel_spec(signal2_local)
                mel_spec = torch.stack([mel_spec, mel_spec2])

        if self.backbone_type == 'FusionModel':
            representation = self.SoundEncoder(signal1_local.unsqueeze(0),
                                               self.environment.mel_spec(signal1_local).unsqueeze(0))

        elif self.backbone_type == 'ResNet':
            representation = self.SoundEncoder(mel_spec.unsqueeze(0)).squeeze(0)

        elif self.backbone_type == 'CNN2D' or self.backbone_type == 'Attention':
            # print(self.__class__)
            representation = self.SoundEncoder(mel_spec.unsqueeze(0)).squeeze(0)

        elif self.backbone_type == 'CNN1D':
            representation = self.SoundEncoder(signal1_local.unsqueeze(0))

        else:
            representation = self.SoundEncoder(torch.fft.rfft(signal1_local, norm='forward').abs())

        return representation

    def get_state_representation(self, state=None) -> (torch.Tensor, torch.Tensor):
        """
        Returns the entire state representation and the signal representation.
        The state representation can be sent through the backbone with a MLP attached to it, or just the backbone.
        The signal representation is just the output from the backbone. This is for monitoring size and expressiveness.

        If the state is not supplied, the state in the environment is taken.
        """

        if state is None:
            # combined_signal_representation = self.get_sound_representation(self.environment.desired_signal,
            #                                                          self.environment.current_signal)
            synth_parameters = self.environment.curr_synth_parameters
            combined_signal_representation = self.get_sound_representation(mel_spec=self.environment.desired_signal_mel_spec)
            # state = [self.environment.desired_signal, self.environment.current_signal, synth_parameters]
        else:
            desired_signal = state[0]
            # current_signal = state[1]
            synth_parameters = state[2]
            # combined_signal_representation = self.get_sound_representation(desired_signal, current_signal)
            combined_signal_representation = self.get_sound_representation(signal1=desired_signal)

        state_representation = [[combined_signal_representation], synth_parameters]
        if self.are_shared_mlp:
            state_representation = self.advanced_representation([combined_signal_representation], synth_parameters)
        return state_representation, combined_signal_representation

    def get_parameters_synth_loss(self, signal_representation=None) -> torch.Tensor:
        """
        Returns parameter loss between the synth and the synth to copy.
        Not used.
        """

        if signal_representation is None:
            signal_representation = self.get_sound_representation(self.environment.current_signal)

        parameters_list, parameters_tensor = self.parameters_regression(signal_representation)
        oscillators_parameters = parameters_list[0]
        filters_parameters = parameters_list[1]
        ADSR_parameters = parameters_list[2]

        ce_loss = torch.nn.CrossEntropyLoss()
        oscillators_loss = 0
        for oscillator_index in range(self.environment.synth.n_oscillators):
            amp, freq = oscillators_parameters[oscillator_index][0]
            waveform_logits = oscillators_parameters[oscillator_index][1]

            oscillator_vector = self.environment.synth.oscillators[oscillator_index].get_module_vector()
            oscillators_loss = oscillators_loss + torch.pow(oscillator_vector[0] - amp, 2)
            oscillators_loss = oscillators_loss + torch.pow(oscillator_vector[1] - freq, 2)

            curr_waveform = torch.argmax(torch.tensor(oscillator_vector[2:], device=self.device, dtype=torch.float))
            oscillators_loss = oscillators_loss + 0.5 * ce_loss(waveform_logits, curr_waveform)

        filters_loss = 0
        for filter_index in range(self.environment.synth.n_oscillators):
            cutoff_freq, resonance = filters_parameters[filter_index][0]
            filter_type_logits = filters_parameters[filter_index][1]

            filter_vector = self.environment.synth.filters[filter_index].get_module_vector()
            filters_loss = filters_loss + torch.pow(filter_vector[0] - cutoff_freq, 2)
            filters_loss = filters_loss + torch.pow(filter_vector[1] - resonance, 2)

            curr_filter_type = torch.argmax(torch.tensor(filter_vector[2:], device=self.device, dtype=torch.float))
            filters_loss = filters_loss + 0.5 * ce_loss(filter_type_logits, curr_filter_type)

        adsr_loss = 0
        attack, decay, sustain, sustain_time, release = ADSR_parameters
        adsr_vector = self.environment.synth.adsr.get_module_vector()
        adsr_loss = adsr_loss + torch.pow(adsr_vector[0] - attack, 2)
        adsr_loss = adsr_loss + torch.pow(adsr_vector[1] - decay, 2)
        adsr_loss = adsr_loss + torch.pow(adsr_vector[2] - sustain, 2)
        adsr_loss = adsr_loss + torch.pow(adsr_vector[3] - sustain_time, 2)
        adsr_loss = adsr_loss + torch.pow(adsr_vector[4] - release, 2)
        return oscillators_loss + filters_loss + adsr_loss

    def init_synth(self, signal, with_noise=False):
        """
        Initialization of the synth parameters using a pretrained model.
        Not used.
        """

        with torch.no_grad():
            signal_features = self.get_sound_representation(signal)
            parameters_list, parameters_tensor = self.parameters_regression(signal_features)
        oscillators_parameters = parameters_list[0]
        filters_parameters = parameters_list[1]
        ADSR_parameters = parameters_list[2]

        for oscillator_index in range(self.environment.synth.n_oscillators):
            amp, freq = oscillators_parameters[oscillator_index][0]
            waveform_logits = oscillators_parameters[oscillator_index][1]
            waveform_probabilities = torch.softmax(waveform_logits, dim=0)
            waveform, _ = self.parameters_regression.oscillators_waveform_regression[
                oscillator_index].sample_based_on_state(waveform_probabilities)

            random_diff = 0
            if with_noise:
                random_diff = np.random.uniform(-500, 500)
            self.environment.synth.oscillators[oscillator_index].freq.value = freq.item() * 4186 + random_diff
            self.environment.synth.oscillators[oscillator_index].amp.value = amp.item()
            self.environment.synth.oscillators[oscillator_index].waveform.value = waveform

        for filter_index in range(self.environment.synth.n_oscillators):
            cutoff_freq, resonance = filters_parameters[filter_index][0]
            filter_type_logits = filters_parameters[filter_index][1]
            filter_type_probabilities = torch.softmax(filter_type_logits, dim=0)
            filter_type, _ = self.parameters_regression.filters_filter_type_regression[
                filter_index].sample_based_on_state(filter_type_probabilities)

            random_diff = 0
            if with_noise:
                random_diff = np.random.uniform(-500, 500)
            self.environment.synth.filters[filter_index].cutoff_freq.value = cutoff_freq.item() * 20000 + random_diff
            self.environment.synth.filters[filter_index].resonance.value = resonance.item() * 1000
            self.environment.synth.filters[filter_index].filter_type.value = filter_type

        attack, decay, sustain, sustain_time, release = ADSR_parameters
        random_diff = 0
        if with_noise:
            random_diff = np.random.uniform(-0.5, 0.5)
        self.environment.synth.adsr.attack.value = attack.item() + random_diff

        random_diff = 0
        if with_noise:
            random_diff = np.random.uniform(-0.5, 0.5)
        self.environment.synth.adsr.decay.value = decay.item() + random_diff

        random_diff = 0
        if with_noise:
            random_diff = np.random.uniform(-0.25, 0.25)
        self.environment.synth.adsr.sustain.value = sustain.item() + random_diff

        random_diff = 0
        if with_noise:
            random_diff = np.random.uniform(-0.5, 0.5)
        self.environment.synth.adsr.sustain_time.value = sustain_time.item() + random_diff

        random_diff = 0
        if with_noise:
            random_diff = np.random.uniform(-0.5, 0.5)
        self.environment.synth.adsr.release.value = release.item() + random_diff

    def to(self, device):
        self.SoundEncoder = self.SoundEncoder.to(device=device)
        if self.are_shared_mlp:
            self.advanced_representation = self.advanced_representation.to(device=device)

    def save_models(self, path):
        model_path = fr'{path}\SoundEncoder.pt'
        torch.save(self.SoundEncoder.state_dict(), model_path)

        if self.are_shared_mlp:
            model_path = fr'{path}\advanced_representation.pt'
            torch.save(self.advanced_representation.state_dict(), model_path)

    def load_models(self, path):
        model_path = fr'{path}\SoundEncoder.pt'
        self.SoundEncoder.load_state_dict(torch.load(model_path))

        if self.are_shared_mlp:
            model_path = fr'{path}\advanced_representation.pt'
            self.advanced_representation.load_state_dict(torch.load(model_path))


class Agent:
    @abstractmethod
    def get_action_from_state_representation(self, state_representation=None, is_optimal=False):
        pass

    @abstractmethod
    def get_action_log_prob(self, action, state_representation, index=None, policy=None):
        pass

    @abstractmethod
    def get_policy_from_state_representation(self, state_representation=None):
        pass


class OptionBasedAgent(Agent):
    def __init__(self, environment: Environment, representation_dim: int, models_dim: int, n_layers, device='cpu',
                 next_agents: list = [], module_name="",
                 max_step_val=None, max_val=None, min_val=None, options: list = [],
                 is_top_agent=False, is_agent=True, starting_temperature=1, is_SRL=False,
                 are_shared_mlp=False):

        self.device = device

        self.representation_dim = representation_dim
        self.models_dim = models_dim
        self.state_dim = environment.curr_synth_parameters.shape[0]
        self.environment = environment
        self.are_shared_mlp = are_shared_mlp

        self.next_agents = next_agents
        self.model_parameters = None

        self.start_time = 0
        self.temperature = starting_temperature
        self.episodes = list()

        self.is_top_agent = is_top_agent
        self._previous_agent = None
        self.module_name = module_name

        self.regression = None
        self.selection = None
        self.state_regression = None
        self.action_regression = None

        for next_agent in self.next_agents:
            next_agent.previous_agent = self

        self.options = list()
        if len(self.next_agents) > 0:
            self.cache_action_log_prob = torch.zeros(1, device=self.device)
            self.options = self.next_agents

        if len(options) > 0:
            self.options = options

        action_dim = 1
        if max_step_val is not None:
            self.regression = Models.ParameterDifferenceRegression(representation_dim=self.representation_dim,
                                                                   n_representations=1,
                                                                   model_dim=models_dim,
                                                                   context_dim=self.state_dim,
                                                                   max_step_val=max_step_val,
                                                                   n_layers=n_layers,
                                                                   is_residual=True,
                                                                   are_shared_mlp=are_shared_mlp).to(torch.float32)
            self.max_val = max_val
            self.min_val = min_val
            action_dim = 1

        elif self.options is not None:
            self.selection = Models.CategoricalChoice(representation_dim=self.representation_dim,
                                                      n_representations=1,
                                                      model_dim=models_dim,
                                                      context_dim=self.state_dim,
                                                      choices=self.options,
                                                      n_layers=n_layers,
                                                      is_residual=True,
                                                      are_shared_mlp=are_shared_mlp).to(torch.float32)
            action_dim = len(options)

        if is_SRL:
            self.state_regression = Models.StateRegression(representation_dim=self.representation_dim,
                                                           action_dim=action_dim,
                                                           model_dim=models_dim,
                                                           context_dim=self.state_dim,
                                                           n_layers=n_layers,
                                                           is_residual=True).to(torch.float32)

            self.action_regression = Models.ActionRegression(representation_dim=self.representation_dim,
                                                             action_dim=action_dim,
                                                             model_dim=models_dim,
                                                             context_dim=self.state_dim,
                                                             n_layers=n_layers,
                                                             is_residual=True).to(torch.float32)

    @property
    def previous_agent(self):
        return self._previous_agent

    @previous_agent.setter
    def previous_agent(self, agent):
        if not isinstance(agent, OptionBasedAgent):
            raise TypeError("Agent must be an agent aswell")
        self._previous_agent = agent

    def get_random_action(self):
        if self.regression is not None:
            value = np.random.uniform(-self.regression.max_step_val, self.regression.max_step_val)
            parameter = torch.tensor(value, device=self.device).unsqueeze(0)
            return parameter, 0

        if self.selection is not None:
            choice_index = np.random.choice(range(self.selection.n_choices))
            choice_index_one_hot = torch.zeros(self.selection.n_choices, device=self.device)
            choice_index_one_hot[choice_index] = 1
            return choice_index, torch.tensor(choice_index, device=self.device, dtype=torch.long), choice_index_one_hot

    def get_policy_from_state_representation(self, state_representation=None) -> (list, torch.Tensor, list):
        """
        Returns policy, termination and logits from policy head of agent.
        When the agent has a regression head:
            policy: mu (value1 * tanh(h))
                    sigma (value2 * sigmoid(h))
                    extremity (abs(tanh(h))
                where h is hidden state after mlp, value1 and value2 can be tuned.

        When the agent has a selection head:
            policy: softmax(h)
            logits: h

        termination: sigmoid(h)
        We regularize the logits and the extremity.
        """

        policy = torch.tensor([1.0])
        termination = torch.tensor(0)
        logits = torch.tensor(0)
        logits_list = list()
        temp_policy = list()

        if len(self.options) > 1:
            if self.are_shared_mlp:
                logits, termination = self.selection(advanced_representation=state_representation)
            else:
                logits, termination = self.selection(state_representation[0], state_representation[1])
            if len(self.next_agents) > 0:
                logits = logits / self.temperature
            logits_list.append(logits)
            policy = torch.softmax(logits, dim=0)
            # if len(self.next_agents) > 0:
            #     print(self.module_name, logits, policy)

        elif self.regression is not None:
            if self.are_shared_mlp:
                policy, termination = self.regression(advanced_representation=state_representation)
            else:
                policy, termination = self.regression(state_representation[0], state_representation[1])

        temp_policy.append(policy)
        return temp_policy, termination, logits_list

    def get_categorical_choice(self, policy, is_optimal=False):
        probabilities = policy
        entropy = - 1 * torch.sum(policy * torch.log(policy))
        selection_idx = self.selection.sample_based_on_state(probabilities,
                                                             is_optimal=is_optimal,
                                                             device=self.device)

        return selection_idx, entropy

    def get_continuous_parameter(self, policy, is_optimal=False):
        [mu, sigma, extremity] = policy
        if is_optimal:
            return mu, extremity, sigma
        parameter = Models.sample_continuous(mu, sigma)
        return parameter, extremity, sigma

    def get_action_from_state_representation(self, state_representation=None, is_optimal=False, policy=None):
        if policy is None:
            policy_list, _, _ = self.get_policy_from_state_representation(state_representation)
            policy = policy_list[0]

        if self.regression is not None:
            parameter, extremity, sigma = self.get_continuous_parameter(policy, is_optimal=is_optimal)
            return parameter, extremity, sigma

        if self.selection is not None:
            choice_idx, entropy = self.get_categorical_choice(policy, is_optimal=is_optimal)
            choice_idx_one_hot = torch.zeros(self.selection.n_choices, device=self.device)
            choice_idx_one_hot[choice_idx] = 1

            return choice_idx, choice_idx_one_hot, entropy

    def get_categorical_choice_log_prob(self, action, state_representation, policy=None):
        if policy is None:
            policy, _, _ = self.get_policy_from_state_representation(state_representation)
        probabilities = policy[0]
        choice_log_prob = Models.get_log_prob_categorical(action, probabilities)
        return choice_log_prob

    def get_continuous_parameter_log_prob(self, action, state_representation, policy=None):
        if policy is None:
            policy, _, _ = self.get_policy_from_state_representation(state_representation)
        mu, sigma, extremity = policy[0]
        parameter_log_prob = Models.get_log_prob_continuous(action, mu, sigma)
        return parameter_log_prob

    def get_action_log_prob(self, action, state_representation, policy=None):
        log_prob = torch.tensor(0)
        if len(self.options) == 1:
            return log_prob

        if self.regression is not None:
            log_prob = self.get_continuous_parameter_log_prob(action, state_representation, policy=policy)

        if self.selection is not None:
            log_prob = self.get_categorical_choice_log_prob(action, state_representation, policy=policy)

        return log_prob

    def get_SRL_predictions(self, action, synth_state, signal1_representation, signal2_representation):
        action_prediction = self.action_regression(signal1_representation, signal2_representation, synth_state)

        if self.regression is not None:
            next_signal_prediction = self.state_regression([signal1_representation], action[0][-2].detach()
                                                           / self.regression.max_step_val, synth_state)

        else:
            next_signal_prediction = self.state_regression([signal1_representation], action[0][-2].detach(), synth_state)

        return next_signal_prediction, action_prediction

    def to(self, device):
        models, names = self.get_models()
        for model in models:
            model = model.to(device)

    def get_models(self, models=[], names=[]):
        for attribute in list(vars(self).keys()):
            value = getattr(self, attribute)
            if isinstance(value, nn.Module):
                models.append(value)
                names.append(attribute)

        if len(self.next_agents) > 0:
            for agent in self.next_agents:
                agent.get_models(models, names)

        return models, names

    def parameters(self, recurse: bool = True):
        models, _ = self.get_models(models=[], names=[])
        for model in models:
            for name, param in model.named_parameters(recurse=recurse):
                yield param

    def save_models(self, path):
        for attribute in list(vars(self).keys()):
            value = getattr(self, attribute)
            if isinstance(value, nn.Module):
                model_path = fr'{path}\{attribute}.pt'
                torch.save(value.state_dict(), model_path)

        if len(self.next_agents) > 0:
            new_path = path + fr'\SubAgent'
            for i, next_agent in enumerate(self.next_agents):
                os.mkdir(new_path + str(i))
                next_agent.save_models(new_path + str(i))

    def load_models(self, path):
        for attribute in list(vars(self).keys()):
            value = getattr(self, attribute)
            if isinstance(value, nn.Module):
                model_path = f'{path}\\{attribute}.pt'
                value.load_state_dict(torch.load(model_path))

        if len(self.next_agents) > 0:
            new_path = path + r'\SubAgent'
            for i, next_agent in enumerate(self.next_agents):
                next_agent.load_models(new_path + str(i))

    def check_stopping(self, task_beta, n_trials=0, is_optimal=False):
        """
            If we stop sub-agent by chance, we check whether it's the top agent.
            If the agent stopped is not a top agent, we continue to sample next agent in hierarchy
        """
        sub_task_stopped = False
        if not self.is_top_agent:
            p = np.random.random()
            task_beta_temp = task_beta.item()
            if n_trials <= 2:
                if (task_beta_temp > 0.5 and is_optimal) or p < task_beta_temp:
                    sub_task_stopped = True
        return sub_task_stopped


class HierarchicalAgent(Manager, Agent):
    def __init__(self, top_agent: OptionBasedAgent, environment: Environment, representation_dim: int, models_dim: int,
                 n_layers, device='cpu', backbone_type='FusionModel', temperature_decrease_rate=0.99,
                 are_shared_mlp=False, is_exploratory=False):
        Manager.__init__(self, environment, representation_dim, models_dim, n_layers, device, backbone_type,
                         are_shared_mlp)
        self.top_agent = top_agent
        self.is_exploratory = is_exploratory
        self.active_agents_indices = list()
        self.current_agent_level = 0
        self.temperature_decrease_rate = temperature_decrease_rate
        self.agents = DFS_agent(top_agent, level2agent=dict())
        self.indices2agent = build_indices(top_agent)
        self.indices2agent_keys = list(self.indices2agent.keys())

    def get_agent_from_indices(self, index=None) -> OptionBasedAgent:
        if index is None:
            index = self.current_agent_level
        agent_temp = self.top_agent
        for idx in range(index):
            next_agent = agent_temp.next_agents[self.active_agents_indices[idx]]
            agent_temp = next_agent
        return agent_temp

    def get_agent_from_action(self, action, index) -> OptionBasedAgent:
        agent_temp = self.top_agent
        for i in range(index):
            next_agent = agent_temp.next_agents[action[i]]
            agent_temp = next_agent
        return agent_temp

    def get_action_from_state_representation(self, state_representation=None, is_optimal=False):
        agent = self.get_agent_from_indices(len(self.active_agents_indices))
        parameter, extremity, entropy = agent.get_action_from_state_representation(state_representation, is_optimal)
        temp_action = self.active_agents_indices.copy()
        temp_action.append(parameter)
        temp_action.append(extremity)
        temp_action.append(entropy)
        return [temp_action]

    def get_action_log_prob(self, action, state_representation, index=None, policy=None):
        if index is None:
            index = self.current_agent_level
        agent = self.get_agent_from_action(action[0], index)
        log_prob = agent.get_action_log_prob(action[0][index], state_representation, policy)
        return log_prob

    def init_agents(self, state_representation=None):
        for level in self.agents.keys():
            for agent in self.agents[level]:
                agent.start_time = 0
        self.active_agents_indices = list(self.indices2agent_keys[1])
        self.current_agent_level = len(self.active_agents_indices)
        active_agent = self.get_agent_from_indices(self.current_agent_level)
        active_agent.start_time = 0
        return

        if state_representation is None:
            state_representation, _ = self.get_state_representation()

        self.current_agent_level = 0
        self.active_agents_indices = list()
        current_agent = self.top_agent
        module_index = torch.tensor(0).to(state_representation.device)
        while len(current_agent.next_agents) > 0:
            next_agent_index, _, _ = current_agent.get_action_from_state_representation(state_representation)
            next_agent = current_agent.next_agents[next_agent_index]
            self.active_agents_indices.append(next_agent_index)
            module_index = self.active_agents_indices[0].squeeze()

            action_log_prob = current_agent.get_action_log_prob(next_agent_index, state_representation)
            current_agent.cache_action_log_prob = action_log_prob.detach()

            current_agent.accumulated_reward = 0
            current_agent.start_time = 0
            current_agent = next_agent
            self.current_agent_level += 1

    def get_policy_from_state_representation(self, state_representation=None):
        policy, termination, _ = self.get_agent_from_indices().get_policy_from_state_representation(state_representation)
        return policy, termination

    def search_agent(self, state_representation, t, logger: Utils.Logger = None, is_optimal=False):
        if t == 0 and self.environment.synth.n_oscillators > 1:
            self.active_agents_indices = list(self.indices2agent_keys[3])
            active_agent = self.get_agent_from_indices(self.current_agent_level)
            active_agent.start_time = 1
            self.top_agent.start_time = 2
            active_agent.previous_agent.start_time = 2
            return

        active_agent = self.get_agent_from_indices(self.current_agent_level)
        agent_for_log = active_agent
        policy, stop_prob, _ = active_agent.get_policy_from_state_representation(state_representation)
        sub_task_stopped = active_agent.check_stopping(stop_prob, 0, is_optimal=is_optimal)
        do_search_agent = sub_task_stopped
        n_trials = 0
        is_direction_up = True
        entropy = None
        while do_search_agent:
            module_name = agent_for_log.module_name
            # Update the agent's policy based on the sum of rewards accumulated during sub-agent's time
            if sub_task_stopped:
                if is_direction_up:
                    active_agent.episodes[-1][3] = True
                self.current_agent_level = self.current_agent_level - 1
                active_agent = self.get_agent_from_indices(self.current_agent_level)
                self.active_agents_indices.pop()
                if not (len(active_agent.episodes) > 0 and active_agent.episodes[-1][1] == t + 1):
                    action_prob = torch.exp(active_agent.cache_action_log_prob)
                    active_agent.episodes.append([active_agent.start_time, t + 1, action_prob, False])
                n_trials += 1
            else:
                active_agent.start_time = t + 1
                if len(active_agent.next_agents) > 0:
                    self.current_agent_level = self.current_agent_level + 1
                    next_agent_index, _, entropy = active_agent.get_action_from_state_representation(state_representation,
                                                                                                     policy=policy[0])
                    next_agent = active_agent.next_agents[next_agent_index]
                    self.active_agents_indices.append(next_agent_index)

                    action_log_prob = active_agent.get_action_log_prob(next_agent_index, state_representation,
                                                                       policy=policy)
                    active_agent.cache_action_log_prob = action_log_prob.detach()
                    active_agent = next_agent
                    is_direction_up = False
                else:
                    do_search_agent = False

            if logger is not None:
                if len(agent_for_log.next_agents) == 0:
                    module_name = agent_for_log.module_name
                logger.what2log_t[f'{module_name} Stop prob vs t'] = stop_prob.item()
                if entropy is not None and len(agent_for_log.next_agents) > 0:
                    logger.what2log_t[f'{module_name}_entropy vs t'] = entropy.item()
                    logger.what2log_t[f'{module_name}_probability 1 vs t'] = policy[0][0].item()

            policy, stop_prob, _ = active_agent.get_policy_from_state_representation(state_representation)
            if is_direction_up:
                if sub_task_stopped and len(active_agent.next_agents) == 1:
                    sub_task_stopped = True
                else:
                    sub_task_stopped = active_agent.check_stopping(stop_prob, n_trials, is_optimal=is_optimal)
            agent_for_log = active_agent

    def to(self, device):
        Manager.to(self, device)
        self.top_agent.to(device)

    def decrease_temperature(self):
        for level in self.agents.keys():
            for agent in self.agents[level]:
                agent.temperature *= self.temperature_decrease_rate
                agent.temperature = max([1, agent.temperature])

    def save_models(self, path):
        super(HierarchicalAgent, self).save_models(path)
        self.top_agent.save_models(path)

    def load_models(self, path):
        super(HierarchicalAgent, self).load_models(path)
        self.top_agent.load_models(path)


class SingleActionAgent(HierarchicalAgent):
    def __init__(self, top_agent: OptionBasedAgent, environment: Environment, representation_dim: int, models_dim: int,
                 n_layers, device='cpu', backbone_type='FusionModel', temperature_decrease_rate=0.99,
                 are_shared_mlp=False):
        super(SingleActionAgent, self).__init__(top_agent, environment, representation_dim, models_dim, n_layers,
                                                device, backbone_type, temperature_decrease_rate, are_shared_mlp)
        self.episodes = list()
        self.module_name = "Single_Agent"

    def get_action_from_state_representation(self, state_representation=None, is_optimal=False):
        actions = DFS_action(self.top_agent, actions=list(), state_representation=state_representation,
                             is_optimal=is_optimal)
        return actions

    def get_action_log_prob(self, action, state_representation, index=None, policy=None):
        log_prob = torch.tensor(0)
        if policy is not None:
            for sub_action, sub_policy in zip(action, policy):
                log_prob = log_prob + DFS_log_prob(self.top_agent, sub_action, state_representation, policy=sub_policy)
        else:
            for sub_action in action:
                log_prob = log_prob + DFS_log_prob(self.top_agent, sub_action, state_representation)
        return log_prob

    def get_policy_from_state_representation(self, state_representation=None):
        policies, logits_list = DFS_policy(self.top_agent, policies=list(), logits_list=list(),
                                           state_representation=state_representation)
        return policies, None, logits_list

    def get_agent_from_action(self, action, index):
        return self

    def to(self, device):
        super(SingleActionAgent, self).to(device)
        self.top_agent.to(device)


class CyclicAgent(HierarchicalAgent):
    def __init__(self, top_agent: OptionBasedAgent, environment: Environment, representation_dim: int, models_dim: int,
                 n_layers, device='cpu', backbone_type='FusionModel', temperature_decrease_rate=0.99,
                 are_shared_mlp=False):
        super(CyclicAgent, self).__init__(top_agent, environment, representation_dim, models_dim, n_layers,
                                          device, backbone_type, temperature_decrease_rate, are_shared_mlp)
        self.current_bottom_agent_index = 0

    def init_agents(self, state_representation=None):
        for agent in self.indices2agent.values():
            agent.accumulated_reward = 0
            agent.start_time = 0

        self.active_agents_indices = list(self.indices2agent_keys[0])
        self.current_agent_level = len(self.active_agents_indices)
        self.current_bottom_agent_index = 0

    def search_agent(self, state_representation, t, logger: Utils.Logger = None, is_optimal=False):
        self.current_bottom_agent_index += 1
        self.current_bottom_agent_index %= len(self.indices2agent)
        self.active_agents_indices = list(self.indices2agent_keys[self.current_bottom_agent_index])
        self.current_agent_level = len(self.active_agents_indices)
        active_agent = self.indices2agent[tuple(self.active_agents_indices)]
        active_agent.start_time = t + 1


class Critic(Manager):
    def __init__(self, environment: Environment, representation_dim: int, models_dim: int, n_layers, device='cpu',
                 backbone_type='FusionModel', exploratory=False, are_shared_mlp=False):
        super(Critic, self).__init__(environment=environment,
                                     representation_dim=representation_dim,
                                     models_dim=models_dim * 5,
                                     n_layers=n_layers,
                                     device=device,
                                     backbone_type=backbone_type,
                                     are_shared_mlp=are_shared_mlp,
                                     activation='Tanh')

        self.state_dim = environment.curr_synth_parameters.shape[0]

        self.ValueEstimation = Models.ValueEstimation(representation_dim=self.representation_dim,
                                                      n_representations=1,
                                                      model_dim=models_dim * 5,
                                                      context_dim=self.state_dim,
                                                      n_layers=n_layers,
                                                      is_residual=False,
                                                      is_deep_wide=False,
                                                      are_shared_mlp=are_shared_mlp,
                                                      scale_factor=environment.max_scaled_val)

        self.target_critic = None

        self.current_value = torch.tensor([0], device=device)

    def set_target_critic(self, other):
        self.target_critic = other

    def get_value_of_state(self, state_representation=None):
        if self.are_shared_mlp:
            self.current_value = self.ValueEstimation(advanced_representation=state_representation)
        else:
            self.current_value = self.ValueEstimation(state_representation[0], state_representation[1])
        return self.current_value

    def copy_weights_to_target(self):
        self.target_critic.ValueEstimation.load_state_dict(self.ValueEstimation.state_dict())
        self.target_critic.SoundEncoder.load_state_dict(self.SoundEncoder.state_dict())
        if self.are_shared_mlp:
            self.target_critic.advanced_representation.load_state_dict(self.advanced_representation.state_dict())

    def get_models(self, models=[], names=[]):
        for attribute in list(vars(self).keys()):
            value = getattr(self, attribute)
            if isinstance(value, nn.Module):
                models.append(value)
                names.append(attribute)
        return models, names

    def to(self, device):
        super(Critic, self).to(device)
        self.ValueEstimation = self.ValueEstimation.to(device)
        if self.are_shared_mlp:
            self.advanced_representation = self.advanced_representation.to(device)

    def save_models(self, path):
        super(Critic, self).save_models(path)
        model_path = fr'{path}\ValueEstimation.pt'
        torch.save(self.ValueEstimation.state_dict(), model_path)

    def load_models(self, path):
        super(Critic, self).load_models(path)
        model_path = fr'{path}\ValueEstimation.pt'
        self.ValueEstimation.load_state_dict(torch.load(model_path))


class SynthRegression(nn.Module):
    def __init__(self, synth: SM.Synthesizer, representation_dim, model_dim, n_layers):
        super(SynthRegression, self).__init__()
        self.n_oscillators = synth.n_oscillators

        self.oscillators_parameters_regression \
            = nn.ModuleList([Models.ParameterRegression(representation_dim, 2, model_dim, n_layers)
                             for i in range(self.n_oscillators)])

        self.oscillators_waveform_regression \
            = nn.ModuleList([Models.CategoricalChoice(representation_dim, 1, 0, synth.available_waves, model_dim, n_layers)
                             for i in range(self.n_oscillators)])

        self.filters_parameters_regression \
            = nn.ModuleList([Models.ParameterRegression(representation_dim, 2, model_dim, n_layers)
                             for i in range(self.n_oscillators)])

        self.filters_filter_type_regression \
            = nn.ModuleList([Models.CategoricalChoice(representation_dim, 1, 0, synth.available_filters, model_dim, n_layers)
                             for i in range(self.n_oscillators)])

        self.ADSR_regression = Models.ParameterRegression(representation_dim, 5, model_dim, n_layers)

    def forward(self, x):
        oscillators_parameters = list()
        filters_parameters = list()

        for i in range(self.n_oscillators):
            oscillator_parameters = list()
            parameters = self.oscillators_parameters_regression[i](x)
            waveform_logits, _ = self.oscillators_waveform_regression[i]([x])

            oscillator_parameters.append(parameters)
            oscillator_parameters.append(waveform_logits)
            oscillators_parameters.append(oscillator_parameters)

            filter_parameters = list()
            parameters = self.filters_parameters_regression[i](x)
            filter_type_logits, _ = self.filters_filter_type_regression[i]([x])

            filter_parameters.append(parameters)
            filter_parameters.append(filter_type_logits)
            filters_parameters.append(filter_parameters)

        ADSR_parameters = self.ADSR_regression(x)
        synth_parameters_list = [oscillators_parameters, filters_parameters, ADSR_parameters]

        synth_parameters_list_for_tensor = list()
        for module_type in synth_parameters_list:
            for module in module_type:
                if isinstance(module, list):
                    for parameters in module:
                        synth_parameters_list_for_tensor.append(parameters)

        synth_parameters_tensor = torch.concat(synth_parameters_list_for_tensor, dim=0)
        return synth_parameters_list, synth_parameters_tensor


def create_trajectory(n_actions, episode: Utils.Episode, agent, env: Environment, logger: Utils.Logger = None, log_data=True,
                      is_optimal=False, play_sound=False, show_spec=False, specs_dir=None, inference_mode=False):
    state_representation, _ = agent.get_state_representation()

    if show_spec:
        plt.imshow(env.current_signal_mel_spec.cpu().numpy())
        plt.show()

    if isinstance(agent, HierarchicalAgent) or isinstance(agent, CyclicAgent) and len(episode.States) <= 0:
        agent.init_agents()

    if show_spec:
        plt.imshow(env.desired_signal_mel_spec.cpu().numpy())
        plt.show()
        if specs_dir is not None:
            plt.savefig(specs_dir + '\\desired.png')

    for i in range(n_actions):
        with torch.no_grad():
            if logger is not None and env.synth.n_oscillators == 1:
                freq_diff = env.synth.oscillators[0].freq.value - env.synth2copy.oscillators[0].freq.value
                logger.what2log_t['Freq diff vs t'] = freq_diff
                amp_diff = env.synth.oscillators[0].amp.value - env.synth2copy.oscillators[0].amp.value
                logger.what2log_t['Amp diff vs t'] = amp_diff
                cutoff_freq_diff = env.synth.filters[0].cutoff_freq.value - env.synth2copy.filters[0].cutoff_freq.value
                logger.what2log_t['Cutoff freq diff vs t'] = cutoff_freq_diff

            action = agent.get_action_from_state_representation(state_representation, is_optimal=is_optimal)
            action_log_prob = agent.get_action_log_prob(action, state_representation)
            episode.Actions.append(action)
            action_prob = torch.exp(action_log_prob)
            episode.ActionProbabilities.append(action_prob)

            reward, next_state = env.step(action, inference_mode=inference_mode)
            if play_sound:
                env.synth2copy.play_sound()
                env.synth.play_sound()

            if show_spec:
                plt.imshow(env.current_signal_mel_spec.cpu().numpy())
                plt.show()

            episode.States.append(next_state)
            episode.Rewards.append(reward)
            state_representation, signal_representation = agent.get_state_representation()
            with torch.no_grad():
                representation_size = torch.sum(torch.abs(signal_representation))

            if logger is not None:
                logger.update_average('immediate_reward', reward - env.reward_offset, add_to_count=True)
                logger.what2log_t['Reward vs t'] = reward - env.reward_offset
                logger.what2log_t['Representation size vs t'] = representation_size
                for sub_action in action:
                    module_agent = agent.top_agent.next_agents[sub_action[0]]
                    parameter_agent = module_agent.next_agents[sub_action[1]]
                    if len(parameter_agent.next_agents) > 0:
                        parameter_agent = parameter_agent.next_agents[sub_action[2]]

                    module_name = parameter_agent.module_name
                    if parameter_agent.regression is not None:
                        logger.what2log_t[f'{module_name} vs t'] = sub_action[-3].item()
                        logger.what2log_t[f'{module_name}_extremity vs t'] = sub_action[-2].item()

                    logger.what2log_t[f'{module_name}_entropy vs t'] = sub_action[-1].item()
                    logger.count_for_average[f'{module_name}_dwell'] += 1
                logger.step += 1

                if log_data and logger.step % 100 == 0:
                    wandb.log(logger.what2log_t, step=logger.step)

            if isinstance(agent, HierarchicalAgent) or isinstance(agent, CyclicAgent):
                agent.get_agent_from_indices().episodes.append([len(episode.Actions) - 1, len(episode.Actions),
                                                                action_prob, False])
                agent.search_agent(state_representation=state_representation,
                                   t=len(episode.Actions) - 1,
                                   logger=logger,
                                   is_optimal=is_optimal)
        terminate = env.get_episode_stopping_condition()
        if terminate:
            episode.terminated = True
            break


def create_hierarchical_agent(env, training_config, model_config, is_single_agent):
    # TODO Fix back to hierarchical
    AgentClass = SingleActionAgent if is_single_agent else HierarchicalAgent
    # max_step_val_multiplier = 3 if is_single_agent else 1
    if env.hierarchy_type == 'module':
        agents = create_module_based_hierarchical_agents(env, training_config, model_config)

    else:
        agents = create_parameter_based_hierarchical_agents(env, training_config, model_config)

    TopAgent = OptionBasedAgent(environment=env,
                                representation_dim=model_config['representation_dim'],
                                models_dim=model_config['models_dim'],
                                n_layers=model_config['n_layers'],
                                device=training_config['device'],
                                next_agents=agents,
                                is_top_agent=True,
                                module_name="TopAgent",
                                are_shared_mlp=model_config["are_shared_mlp"],
                                starting_temperature=training_config["starting_temperature"])

    Manager = AgentClass(TopAgent,
                         environment=env,
                         representation_dim=model_config['representation_dim'],
                         models_dim=model_config['models_dim'],
                         n_layers=model_config['n_layers'],
                         device=training_config['device'],
                         backbone_type=model_config['Backbone'],
                         temperature_decrease_rate=training_config['temperature_decrease_rate'],
                         are_shared_mlp=model_config["are_shared_mlp"],)
    return Manager


def create_module_based_hierarchical_agents(env, training_config, model_config):
    module_agents = list()
    for module in env.synth.editable_modules:
        parameters_agents = list()
        for parameter in module.editable_parameters:
            max_step_val = None
            min_val = None
            max_val = None
            choices = list()
            if parameter == 'freq':
                max_step_val = 800
                min_val = 27.5
                max_val = 4186

            elif parameter == 'amp':
                max_step_val = 0.1 if training_config['action_scale'] == 'linear' else 8
                min_val = 0.001
                max_val = 1

            elif parameter == 'waveform':
                choices = env.synth.available_waves

            elif parameter == 'cutoff_freq':
                max_step_val = 500 / env.synth.n_oscillators
                # max_step_val = 500 if training_config['action_scale'] == 'linear' else 8
                min_val = 20
                max_val = 20000

            elif parameter == 'resonance':
                max_step_val = 20 if training_config['action_scale'] == 'linear' else 2.9
                min_val = 0.1
                max_val = 100

            elif parameter == 'filter_type':
                choices = env.synth.available_filters

            else:
                max_step_val = 0.2 if training_config['action_scale'] == 'linear' else 8
                min_val = 0.0001
                max_val = 1

            Agent = create_range_agents(env, max_step_val, min_val, max_val, choices,
                                        parameter, module.name, training_config, model_config)
            parameters_agents.append(Agent)
        module_agent = OptionBasedAgent(environment=env,
                                        representation_dim=model_config['representation_dim'],
                                        models_dim=model_config['models_dim'],
                                        n_layers=model_config['n_layers'],
                                        device=training_config['device'],
                                        next_agents=parameters_agents,
                                        module_name=module.name,
                                        are_shared_mlp=model_config["are_shared_mlp"],
                                        starting_temperature = training_config["starting_temperature"])
        module_agents.append(module_agent)
    return module_agents


def create_parameter_based_hierarchical_agents(env, training_config, model_config):
    parameters_agents = list()
    adsr_list_temp = [env.synth.adsr]
    for current_module_list in [env.synth.oscillators, env.synth.filters, adsr_list_temp]:
        for parameter in current_module_list[0].editable_parameters:
            parameter_agents = list()
            for module in current_module_list:
                max_step_val = None
                min_val = None
                max_val = None
                choices = list()
                if parameter == 'freq':
                    max_step_val = 400
                    min_val = 27.5
                    max_val = 4186

                elif parameter == 'amp':
                    max_step_val = 0.1 if training_config['action_scale'] == 'linear' else 8
                    min_val = 0.001
                    max_val = 1

                elif parameter == 'waveform':
                    choices = env.synth.available_waves

                elif parameter == 'cutoff_freq':
                    max_step_val = 500 / env.synth.n_oscillators
                    # max_step_val = 500 if training_config['action_scale'] == 'linear' else 8
                    min_val = 20
                    max_val = 20000

                elif parameter == 'resonance':
                    max_step_val = 20 if training_config['action_scale'] == 'linear' else 2.9
                    min_val = 0.1
                    max_val = 100

                elif parameter == 'filter_type':
                    choices = env.synth.available_filters

                else:
                    max_step_val = 0.2 if training_config['action_scale'] == 'linear' else 8
                    min_val = 0.0001
                    max_val = 1
                Agent = create_range_agents(env, max_step_val, min_val, max_val, choices,
                                            module.name, parameter, training_config, model_config)
                parameter_agents.append(Agent)

            parameter_agent = OptionBasedAgent(environment=env,
                                               representation_dim=model_config['representation_dim'],
                                               models_dim=model_config['models_dim'],
                                               n_layers=model_config['n_layers'],
                                               device=training_config['device'],
                                               next_agents=parameter_agents,
                                               module_name=parameter,
                                               are_shared_mlp=model_config["are_shared_mlp"],
                                               starting_temperature = training_config["starting_temperature"])
            parameters_agents.append(parameter_agent)
    return parameters_agents


def create_range_agents(env, max_step_val, min_val, max_val, options, parameter, module_name,
                        training_config, model_config):
    # If the agent is exploratory then it is separated to 2 agents with different step sizes.
    if len(options) != 0 or not training_config['exploratory']:
        Agent = OptionBasedAgent(environment=env,
                                 representation_dim=model_config['representation_dim'],
                                 models_dim=model_config['models_dim'],
                                 n_layers=model_config['n_layers'],
                                 device=training_config['device'],
                                 max_step_val=max_step_val,
                                 min_val=min_val,
                                 max_val=max_val,
                                 options=options,
                                 module_name=f'{module_name}_{parameter}',
                                 is_SRL=training_config['SRL'],
                                 are_shared_mlp=model_config["are_shared_mlp"],
                                 starting_temperature = training_config["starting_temperature"])
    else:
        multiplier = 2
        small_range_agent = OptionBasedAgent(environment=env,
                                             representation_dim=model_config['representation_dim'],
                                             models_dim=model_config['models_dim'],
                                             n_layers=model_config['n_layers'],
                                             device=training_config['device'],
                                             max_step_val=max_step_val / 2,
                                             min_val=min_val,
                                             max_val=max_val,
                                             options=options,
                                             module_name=f'{module_name}_{parameter}_small',
                                             is_SRL=training_config['SRL'],
                                             are_shared_mlp=model_config["are_shared_mlp"],
                                             starting_temperature = training_config["starting_temperature"])

        large_range_agent = OptionBasedAgent(environment=env,
                                             representation_dim=model_config['representation_dim'],
                                             models_dim=model_config['models_dim'],
                                             n_layers=model_config['n_layers'],
                                             device=training_config['device'],
                                             max_step_val=max_step_val * multiplier,
                                             min_val=min_val,
                                             max_val=max_val,
                                             options=options,
                                             module_name=f'{module_name}_{parameter}_large',
                                             is_SRL=training_config['SRL'],
                                             are_shared_mlp=model_config["are_shared_mlp"],
                                             starting_temperature = training_config["starting_temperature"])

        range_agents = [small_range_agent, large_range_agent]

        Agent = OptionBasedAgent(environment=env,
                                 representation_dim=model_config['representation_dim'],
                                 models_dim=model_config['models_dim'],
                                 n_layers=model_config['n_layers'],
                                 device=training_config['device'],
                                 next_agents=range_agents,
                                 module_name=f'{module_name}_{parameter}',
                                 are_shared_mlp=model_config["are_shared_mlp"],
                                 starting_temperature = training_config["starting_temperature"])
    return Agent


def initialize_env(synth_config_dir, environment_config_dir) -> Environment:
    with open(synth_config_dir) as f:
        synth_config = dict(json.load(f))

    with open(environment_config_dir) as f:
        environment_config = dict(json.load(f))

    max_state_val = 0
    for reward_type, reward_weight in zip(environment_config['reward_type'],
                                          environment_config['reward_weight']):
        if reward_type == 'similarity':
            max_state_val += reward_weight * 1

        elif reward_type == 'earth_mover':
            max_state_val += -1 * reward_weight * 128

        elif reward_type == 'diff_similarity':
            max_state_val += reward_weight * 1

        elif reward_type == 'peak_diff':
            max_state_val += -1 * reward_weight * 1

        elif reward_type == 'parameter_loss':
            n_oscillator_parameters = len(synth_config['oscillator_editable_parameters']) \
                if 'oscillator_editable_parameters' in synth_config.keys() else 0
            n_filter_parameters = len(synth_config['filter_editable_parameters']) \
                if 'filter_editable_parameters' in synth_config.keys() else 0
            n_adsr_parameters = len(synth_config['ADSR_editable_parameters']) \
                if 'ADSR_editable_parameters' in synth_config.keys() else 0

            n_parameters = (n_oscillator_parameters * synth_config['n_oscillators'] +
                            n_filter_parameters * synth_config['n_oscillators'] +
                            n_adsr_parameters)
            max_state_val += -1 * reward_weight * n_parameters

    scaled_value = 10
    env = Environment(synth_config,
                      hierarchy_type=environment_config['hierarchy_type'],
                      is_dynamic_synth=environment_config['is_dynamic_synth'],
                      randomize_time=environment_config['randomize_time'],
                      device=environment_config['device'],
                      reward_type=environment_config['reward_type'],
                      reward_weight=environment_config['reward_weight'],
                      reward_offset=environment_config['reward_offset'],
                      starting_tolerance=environment_config['starting_tolerance'],
                      ending_tolerance=environment_config['ending_tolerance'],
                      edge_penalty=environment_config['edge_penalty'],
                      action_scale=environment_config['action_scale'],
                      max_state_val=max_state_val,
                      scale_factor=max_state_val / scaled_value)

    return env


def initialize_agents(env, training_config_dir, model_config_dir, is_single_agent) -> (OptionBasedAgent, Critic):
    with open(training_config_dir) as f:
        training_config = dict(json.load(f))

    with open(model_config_dir) as f:
        model_config = dict(json.load(f))

    agent = create_hierarchical_agent(env=env, training_config=training_config, model_config=model_config,
                                      is_single_agent=is_single_agent)

    critic = Critic(environment=env,
                    representation_dim=model_config['representation_dim'],
                    models_dim=model_config['models_dim'],
                    n_layers=model_config['n_layers'],
                    device=training_config['device'],
                    backbone_type=model_config['Backbone'],
                    are_shared_mlp=model_config["are_shared_mlp"])

    target_critic = Critic(environment=env,
                           representation_dim=model_config['representation_dim'],
                           models_dim=model_config['models_dim'],
                           n_layers=model_config['n_layers'],
                           device=training_config['device'],
                           backbone_type=model_config['Backbone'],
                           are_shared_mlp=model_config["are_shared_mlp"])

    if model_config["are_shared_backbone"]:
        critic.SoundEncoder = agent.SoundEncoder
        # critic.ParametersEncoder = agent.ParametersEncoder

    return agent, critic, target_critic


def DFS_agent_parameters(agent, level2model: dict = {}, level: int = 0):
    if level not in level2model:
        level2model[level] = list()

    if len(agent.next_agents) > 0:
        level2model[level].append((agent.selection, "agent_choice", agent))
        for next_agent in agent.next_agents:
            DFS_agent_parameters(next_agent, level2model, level + 1)
    else:
        if agent.regression is not None:
            level2model[level].append((agent.regression, "regression", agent))
        if agent.selection is not None:
            level2model[level].append((agent.selection, "selection", agent))
        if agent.action_regression is not None:
            level2model[level].append((agent.action_regression, "action_regression", agent))
        if agent.state_regression is not None:
            level2model[level].append((agent.state_regression, "state_regression", agent))

    if agent.is_top_agent:
        return level2model


def DFS_agent(agent: OptionBasedAgent, level2agent: dict = {}, level: int = 0):
    if level not in level2agent:
        level2agent[level] = list()

    if len(agent.next_agents) > 0:
        level2agent[level].append(agent)
        for next_agent in agent.next_agents:
            DFS_agent(next_agent, level2agent, level + 1)
    else:
        level2agent[level].append(agent)

    if agent.is_top_agent:
        return level2agent


def DFS_action(agent: OptionBasedAgent, actions: list = [], action: list = [], state_representation=None,
               is_optimal=False):
    if len(agent.next_agents) > 0:
        for agent_index, next_agent in enumerate(agent.next_agents):
            action_temp = action.copy()
            action_temp.append(torch.tensor(agent_index))
            DFS_action(next_agent, actions, action_temp, state_representation, is_optimal)
    else:
        action_temp = action.copy()
        parameter, extremity, entropy = agent.get_action_from_state_representation(state_representation,
                                                                                   is_optimal=is_optimal)
        action_temp.append(parameter)
        action_temp.append(extremity)
        action_temp.append(entropy)
        actions.append(action_temp)

    if agent.is_top_agent:
        return actions


def DFS_policy(agent: OptionBasedAgent, policies: list = [], logits_list: list = [], state_representation=None):
    if len(agent.next_agents) > 0:
        for agent_index, next_agent in enumerate(agent.next_agents):
            DFS_policy(next_agent, policies, logits_list, state_representation)
    else:
        policy, _, logits = agent.get_policy_from_state_representation(state_representation)
        policies.append(policy[0])
        logits_list.append(logits)

    if agent.is_top_agent:
        return policies, logits_list


def DFS_log_prob(agent: OptionBasedAgent, action, state_representation, index=0, policy=None):
    if len(agent.next_agents) > 0:
        next_agent = agent.next_agents[action[index].item()]
        log_prob = DFS_log_prob(next_agent, action, state_representation, index + 1)
    else:
        log_prob_agent = agent.get_action_log_prob(action[index], state_representation, index, policy)
        return log_prob_agent
    return log_prob


def build_indices(agent: OptionBasedAgent, current_index: int = 0, current_list: list = [], indices2agent: dict = {}):
    # Include the current node's index
    current_list.append(torch.tensor(current_index, device='cuda').unsqueeze(0))

    # If it's a leaf node, return
    if len(agent.next_agents) == 0:
        # Add the current path to the result
        indices2agent[tuple(current_list[1:])] = agent

    for i, next_agent in enumerate(agent.next_agents):
        build_indices(next_agent, i, current_list, indices2agent)

    # Backtrack: remove the current node's value
    current_list.pop()
    return indices2agent


def test_model(test_set, n_actions, env: Environment, agent, play_sound=False, save_sound=False, show_spec=False, t=0,
               sound_dir: str = ""):
    avg_loss = 0
    avg_freq_diff = 0
    avg_cutoff_freq_diff = 0
    rmse = 0
    count_play = 0
    play_sound_local = False
    show_spec_local = False
    save_sound_local = False
    save_arrows_flag = False
    point_dir = None
    n_test = 1000
    n_graph = 100
    desired_points = list()
    final_points = list()
    for i, point in enumerate(test_set[:n_test]):
        count_play += 1

        episode = Utils.Episode()
        env.synth2copy.init_synth_from_list(point[0])
        env.synth.init_synth_from_list(point[1])
        env.update_signal(3)

        state = env.get_state()
        episode.States.append(state)

        if count_play == 8:
            play_sound_local = True if play_sound else False
            save_sound_local = True if save_sound else False
            show_spec_local = True if show_spec else False
                # point_dir = specs_dir + f'\\point_{i}'
                # os.mkdir(point_dir)
            count_play = 0

        create_trajectory(n_actions, episode, agent, env, is_optimal=True,
                          play_sound=play_sound_local, show_spec=show_spec_local, specs_dir=point_dir,
                          inference_mode=True)
        if save_sound_local:
            write(fr'{sound_dir}/{i}desired.wav', env.synth.sample_rate, env.desired_signal.cpu().numpy())
            write(fr'{sound_dir}/{i}current.wav', env.synth.sample_rate, env.current_signal.cpu().numpy())

        # max_val_d = torch.max(env.desired_signal_mel_spec).item()
        # max_val_c = torch.max(env.current_signal_mel_spec).item()
        # # if max_val_c > 10 or max_val_c < 0.005:
        # print('current', env.synth.oscillators[0].freq.value, env.synth.oscillators[0].waveform.value, max_val_c)
        # print(env.synth.filters[0].cutoff_freq.value, env.synth.oscillators[0].phase.value)
        #     # plt.imshow(env.current_signal_mel_spec.cpu().numpy() * 15)
        #     # plt.show()
        #
        # # if max_val_d > 10 or max_val_d < 0.005:
        # print('desired', env.synth2copy.oscillators[0].freq.value, env.synth2copy.oscillators[0].waveform.value, max_val_d)
        # print(env.synth2copy.filters[0].cutoff_freq.value)
            # plt.imshow(env.desired_signal_mel_spec.cpu().numpy() * 15)
            # plt.show()

        # sound_rep = agent.get_sound_representation(env.desired_signal)
        # parameters, _ = parameter_regression(sound_rep)

        # for i, oscillator in enumerate(env.synth.oscillators):
        #     oscillator.freq.value = parameters[0][i][0][0].item() * oscillator.freq.max_val
        #
        # env.update_signal(1)

        env.update_value()
        avg_loss += env.curr_value

        show_spec_local = False
        play_sound_local = False
        save_sound_local = False

        func = lambda x: x.freq.value

        oscillators1 = sorted(env.synth.oscillators.copy(), key=func)
        oscillators2 = sorted(env.synth2copy.oscillators.copy(), key=func)

        avg_freq_diff += sum([abs(o1.freq.value - o2.freq.value) for o1, o2 in zip(oscillators1, oscillators2)])
        # avg_cutoff_freq_diff += sum([abs(o1.output_module.cutoff_freq.value - o2.output_module.cutoff_freq.value)
        #                              for o1, o2 in zip(oscillators1, oscillators2)])

        final_points.append([o1.freq.value for o1 in env.synth.oscillators.copy()])
        desired_points.append([o2.freq.value for o2 in env.synth2copy.oscillators.copy()])

        rmse += torch.sqrt(torch.mean(torch.pow(env.desired_signal_mel_spec[:, 50: 150] - env.current_signal_mel_spec[:, 50: 150], 2)))
        episode.clear()

    if save_arrows_flag:
        fig, ax = plt.subplots()

        p_f = ax.scatter(*zip(*final_points[:n_graph]), c='blue', marker='o', s=10, zorder=3)
        p_d = ax.scatter(*zip(*desired_points[:n_graph]), c='red', marker='o', s=10, zorder=2)

        x_diff = [p2[0] - p1[0] for p1, p2 in zip(desired_points[:n_graph], final_points[:n_graph])]
        y_diff = [p2[1] - p1[1] for p1, p2 in zip(desired_points[:n_graph], final_points[:n_graph])]

        ax.quiver(*zip(*desired_points[:n_graph]), x_diff, y_diff, angles='xy', scale_units='xy', scale=1, width=0.0025)

        ax.set(xlabel='Oscillator 1 frequency', ylabel='Oscillator 2 frequency', title='Desired to Final point')

        current_directory_path = os.getcwd()
        plt.savefig(fr'{current_directory_path}/RL_Synth/Training/Arrows/{6}_{t}.png')
    print(i)
    return avg_loss * env.scale_factor / n_test, avg_freq_diff / n_test, avg_cutoff_freq_diff / n_test, rmse / n_test
