#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/02/2023

@author: Moshe Laufer, Noy Uzrad, Sagiv Ferster
"""

from abc import ABC, abstractmethod
import torch
import math
from torchaudio.functional.filtering import lowpass_biquad, highpass_biquad, band_biquad
import numpy as np
import simpleaudio as sa
import soundfile as sf
import random

has_vmap = False

PI = math.pi
TWO_PI = 2 * PI

idx2waveform = {0: 'sine', 1: 'square', 2: 'triangle', 3: 'sawtooth'}
waveform2idx = {'sine': 0, 'square': 1, 'triangle': 2, 'sawtooth': 3}

idx2filter = {0: 'lowpass', 1: 'highpass', 2: 'bandpass'}
filter2idx = {'lowpass': 0, 'highpass': 1, 'bandpass': 2}


def indent_module_text(module_str, indentation):
    new_str = str()
    module_str_split = module_str.split('\n')
    for line in module_str_split[:-1]:
        new_str += ' ' * indentation + line + '\n'
    return new_str + ' ' * indentation + module_str_split[-1]


class Synthesizer:
    def __init__(self, device, sample_rate, signal_duration,
                 n_oscillators, available_module_types, editable_module_types,
                 available_waves, available_filters, available_parameters, editable_parameters):
        """ Initialize the synth with a predefined number of oscillators """

        self._n_oscillators = n_oscillators

        """ Oscillators """
        self.oscillators = [FMOscillator(is_active=True, device=device, waveform='sine', name=f'Oscillator{i}',
                                         available_parameters=available_parameters['oscillator_available_parameters'],
                                         editable_parameters=editable_parameters['oscillator_editable_parameters'],
                                         available_waveforms=available_waves)
                            for i in range(n_oscillators)]

        """ Filter for each oscillator """
        self.filters = [Filter(input_module=None, output_module=None, is_active=True, device=device, name=f'Filter{i}',
                               cutoff_freq=0.0, resonance=0.0001,
                               available_parameters=available_parameters['filter_available_parameters'],
                               editable_parameters=editable_parameters['filter_editable_parameters'],
                               available_filter_types=available_filters)
                        for i in range(n_oscillators)]

        # self.amplifier = Amplifier(input_module=None, output_module=None, is_active=True, device=device, name='Amplifier',
        #                            amp_factor=1, modulator=None)

        """ ADSR """
        self.adsr = ADSR(input_module=None, output_module=None, is_active=True, device=device, name='ADSR',
                         attack=0, decay=0, sustain=0, sustain_time=0, release=0, sample_rate=sample_rate,
                         available_parameters=available_parameters['ADSR_available_parameters'],
                         editable_parameters=editable_parameters['ADSR_editable_parameters'])

        # self.amp_adsr = ADSR(input_module=self.amplifier, output_module=None, is_active=True, device=device, name='amp_ADSR',
        #                      attack=0, decay=0, sustain=0, sustain_time=0, release=0, sample_rate=sample_rate)

        """ Mixer """
        self.mixer = Mixer(list(), None, is_active=True, device=device, name='Mixer')

        """ 
            Final module that outputs the sound.
            Can be either one of the modules.
        """
        self._output = Output(input_module=None, output_module=None, is_active=True, device=device, name='Output')

        self.sample_rate = sample_rate
        self.signal_duration = signal_duration

        self.t = torch.linspace(0, self.signal_duration,
                                steps=int(self.sample_rate * self.signal_duration),
                                device=device,
                                requires_grad=False)

        self.available_waves = available_waves
        self.available_filters = available_filters

        self.available_module_types = self.get_module_classes(available_module_types)
        self.editable_module_types = self.get_module_classes(editable_module_types)

        self.modules = self.get_modules()
        self.available_modules = self.get_constrained_modules(editable=False)
        self.editable_modules = self.get_constrained_modules(editable=True)

    def connect_modules(self, module_as_input, module_as_output):
        """
            Connect 2 synth modules
            Can send strings of the module names and also the module objects themself.
        """

        m1 = module_as_input
        if not issubclass(type(module_as_input), SynthModule):
            m1 = getattr(self, module_as_input)

        m2 = module_as_output
        if not issubclass(type(module_as_output), SynthModule):
            m2 = getattr(self, module_as_output)

        if isinstance(m1.output_module, Mixer):
            m1.output_module.disconnect_input(m1)

        # if m1.output_module is not None:
        #     m1.disconnect_output()

        m2.connect_input(m1)

    def __call__(self, t):
        return self._output(t)

    def __str__(self):
        sound_generator_text = "Output: " + str(self._output)

        oscillators_str = [str(x) for x in self.oscillators]
        oscillators_text = "Oscillators: [\n"
        for oscillator in oscillators_str:
            oscillators_text += indent_module_text(oscillator, 2) + ",\n"
        oscillators_text += ']'

        filters_str = [str(x) for x in self.filters]
        filters_text = "Filters: [\n"
        for filter in filters_str:
            filters_text += indent_module_text(filter, 2) + ",\n"
        filters_text += ']'

        adsr_text = str(self.adsr)

        mixer_text = str(self.mixer)

        return sound_generator_text + '\n' + oscillators_text + '\n' + filters_text + '\n' + adsr_text \
               + '\n' + mixer_text

    def get_module_from_synth(self, key):
        if key is None:
            return None
        if key == 'Output':
            return self._output
        if key == 'Mixer':
            return self.mixer
        elif key == 'ADSR':
            return self.adsr
        elif key[0] == 'O':
            return self.oscillators[int(key[1])]
        elif key[0] == 'F':
            return self.filters[int(key[1])]

    @property
    def n_oscillators(self):
        return self._n_oscillators

    @n_oscillators.setter
    def n_oscillators(self, value):
        raise "Can't change the number of oscillators, deactivate or disconnect them instead"

    @staticmethod
    def get_module_classes(module_types):
        available_types = list()
        for t in module_types:
            if t == 'Output':
                available_types.append(Output)
            elif t == 'Mixer':
                available_types.append(Mixer)
            elif t == 'ADSR':
                available_types.append(ADSR)
            elif t == 'O':
                available_types.append(FMOscillator)
            elif t == 'F':
                available_types.append(Filter)

        return available_types

    def get_modules(self):
        module_list = list()
        for module_name in list(vars(self).keys()):
            module = getattr(self, module_name)
            if isinstance(module, list):  # In case of filters, oscillators, etc...
                for sub_module in module:
                    if not isinstance(sub_module, SynthModule):
                        break
                    module_list.append(sub_module)
            elif isinstance(module, SynthModule):
                module_list.append(module)
        return module_list

    def get_constrained_modules(self, editable=True):
        module_list = list()
        module_types = self.editable_module_types if editable else self.available_module_types
        for module_type in module_types:
            for module_name in list(vars(self).keys()):
                if module_name == 'available_modules' or module_name == 'modules':
                    continue
                module = getattr(self, module_name)
                if isinstance(module, list):  # In case of filters, oscillators, etc...
                    for sub_module in module:
                        if isinstance(sub_module, module_type):
                            module_list.append(sub_module)
                elif isinstance(module, module_type):
                    module_list.append(module)
        return module_list

    def randomize_parameters(self):
        for module in self.available_modules:
            module.randomize_module()

    def randomize_flow(self):
        available_input_modules = self.available_modules.copy()
        available_input_modules.remove(self._output)
        available_ouput_modules = list()
        available_ouput_modules.append(self._output)

        for module in self.available_modules:
            module.disconnect_inputs()

        while available_input_modules:
            input_module = np.random.choice(available_input_modules)
            output_module = np.random.choice(available_ouput_modules)

            available_input_modules.remove(input_module)
            if not isinstance(output_module, Mixer):
                available_ouput_modules.remove(output_module)
            available_ouput_modules.append(input_module)

            self.connect_modules(module_as_input=input_module, module_as_output=output_module)

    def check_connection(self, input_module, output_module):
        is_ok = True
        if input_module is output_module:  # Can't connect a module to itself
            is_ok = False
        if input_module.output_module is output_module.input_module:  # Can't make loops
            is_ok = False
        if isinstance(input_module, Mixer) and output_module in input_module.input_modules:
            is_ok = False
        if isinstance(input_module, Output):  # Can't pick Output module as input to another module
            is_ok = False
        return is_ok

    def get_synth_parameters(self):
        parameter_list = list()
        for module in self.available_modules:
            parameter_list = parameter_list + module.get_module_parameters()
        return parameter_list

    def init_synth_from_list(self, parameter_list):
        parameter_list_idx = 0
        for module in self.available_modules:
            n_parameters = len(module.available_parameters)
            for i in range(n_parameters):
                parameter_object = module.__getattribute__(module.available_parameters[i])
                if len(parameter_object.options) != 0:
                    parameter_object.value = parameter_object.options[parameter_list[parameter_list_idx]]
                else:
                    parameter_object.value = parameter_list[parameter_list_idx] * parameter_object.max_val
                parameter_list_idx += 1

    def play_sound(self):
        play_obj = sa.play_buffer(audio_data=self(self.t).cpu().numpy(),
                                  num_channels=1,
                                  bytes_per_sample=4,
                                  sample_rate=self.sample_rate)
        play_obj.wait_done()

    def save_sound(self):
        sf.write('synth_sound.wav', self(self.t).cpu().numpy(), self.sample_rate)


class ModuleParameter(ABC):
    def __init__(self, value, min_val=0.0, max_val=1.0, options=[]):
        self._min_val = min_val
        self._max_val = max_val
        self._options = options
        self._idx2option = {i: option for i, option in enumerate(options)}
        self._option2idx = {option: i for i, option in enumerate(options)}
        self._value = value
        self.value_index = None

    @property
    def min_val(self):
        return self._min_val

    @min_val.setter
    def min_val(self, new_min_val):
        self._min_val = new_min_val

    @property
    def max_val(self):
        return self._max_val

    @max_val.setter
    def max_val(self, new_max_val):
        self._max_val = new_max_val

    @property
    def options(self):
        return self._options

    @property
    def option2idx(self):
        return self._option2idx

    @property
    def idx2option(self):
        return self._idx2option

    def add_option(self, option):
        self._options.append(option)

    def remove_option(self, option):
        self._options.remove(option)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if len(self._options) == 0:
            new_value = self.cut_parameter(new_value)
        else:
            if new_value not in self._options:
                new_value = self._options[0]
                self.value_index = self._option2idx[new_value]
        self._value = new_value

    def cut_parameter(self, value):
        value = max([value, self.min_val])
        value = min([value, self.max_val])
        return value

    def randomize_parameter(self):
        if len(self._options) == 0:
            new_value = np.random.uniform(self.min_val, self.max_val)
        else:
            new_value = random.choice(self._options)
        self._value = new_value


class SynthModule(ABC):
    """
        Abstract Synth module class.
        Synth module can be connected to other synth modules.
        Processes sound according to implemented logic.
    """
    def __init__(self, input_module, output_module, is_active=True, device='cpu', name='', sample_rate=44100, modulator=None):
        self.name = name
        self.input_module = input_module
        self.output_module = output_module
        self.signal = None
        self.is_active = is_active
        self.device = device
        self.sample_rate = sample_rate
        self.modulator = modulator
        self.parameters = list()
        self.available_parameters = list()
        self.editable_parameters = list()

    def __call__(self, t) -> torch.Tensor:
        """
            When calling the synth module we call the implemented method: _generate_wave.
            We set and return the signal generated.
        """
        if self.is_active:
            self.signal = self._generate_wave(t)
            return self.signal
        else:
            return torch.zeros_like(t)

    def _generate_wave(self, t):
        """
            Implemented method for wave generation.
            Args:
                self: Self object. Acts as modulating signal
                t: torch tensor

            Returns:
                A torch with the constructed FM signal

            Raises:
                ValueError: Provided variables are out of range
        """
        return torch.zeros_like(self.input_module(t))

    def connect_input(self, other):
        self.__setattr__('input_module', other)
        other.__setattr__('output_module', self)

    def disconnect_inputs(self):
        self.input_module = None
        if self.input_module is not None:
            self.input_module.output_module = None

    def disconnect_output(self):
        self.output_module = None
        if self.output_module is not None:
            self.output_module.input_module = None

    def deactivate(self):
        self.is_active = False

    def activate(self):
        self.is_active = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def input_module(self):
        return self._input_module

    @input_module.setter
    def input_module(self, new_module):
        assert issubclass(type(new_module), SynthModule) or new_module is None, \
            "The input module must be a synth module"
        self._input_module = new_module

    @property
    def output_module(self):
        return self._output_module

    @output_module.setter
    def output_module(self, new_module):
        assert issubclass(type(new_module), SynthModule) or new_module is None, \
            "The input module must be a synth module"
        self._output_module = new_module

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        assert type(value) is bool, "is_active must be of boolean type"
        self._is_active = value

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_device):
        self._device = new_device

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    def __str__(self):
        name_text = str(self.name)

        input_str = self.input_module.name if self.input_module is not None else "None"
        input_text = f"Input: " + input_str

        output_str = self.output_module.name if self.output_module is not None else "None"
        output_text = f"Output: " + output_str
        return name_text + ': \n' + f"  {input_text}\n  {output_text}"

    def get_module_vector(self):
        parameter_list = list()
        for parameter in self.available_parameters:
            module_parameter_object = self.__getattribute__(parameter)
            if len(module_parameter_object.options) != 0:
                one_hot_vector = [0 for _ in module_parameter_object.options]
                one_hot_vector[module_parameter_object.option2idx[module_parameter_object.value]] = 1
                parameter_list = parameter_list + one_hot_vector
            else:
                parameter_list.append(module_parameter_object.value / module_parameter_object.max_val)
        return parameter_list

    def get_module_parameters(self):
        parameter_list = list()
        for parameter in self.available_parameters:
            module_parameter_object = self.__getattribute__(parameter)
            if len(module_parameter_object.options) != 0:
                parameter_list.append(module_parameter_object.option2idx[module_parameter_object.value])
            else:
                parameter_list.append(module_parameter_object.value / module_parameter_object.max_val)
        return parameter_list

    def randomize_module(self):
        for parameter in self.available_parameters:
            parameter_object = self.__getattribute__(parameter)
            parameter_object.randomize_parameter()


class FMOscillator(SynthModule):
    def __init__(self, input_module=None, output_module=None, is_active=False, device='cpu', name='', sample_rate=44100,
                 amp=0.0, freq=0.0, waveform=None, phase=0.0, modulation_factor=1,
                 available_parameters=['freq'], editable_parameters=['freq'],
                 available_waveforms=['sawtooth']):
        super(FMOscillator, self).__init__(input_module, output_module, is_active, device, name, sample_rate)

        if None in (amp, freq, waveform):
            raise ValueError("Not all necessary attributes were given: [amplitude, frequency, waveform]")

        self._amp = ModuleParameter(value=amp, min_val=0.001, max_val=1)
        self._freq = ModuleParameter(value=freq, min_val=27.5, max_val=4186)
        self._waveform = ModuleParameter(value=waveform, options=available_waveforms)
        self._phase = ModuleParameter(value=0, min_val=0, max_val=TWO_PI)
        self.modulation_factor = ModuleParameter(value=1, min_val=1, max_val=1)
        self.parameters = ['freq', 'amp', 'waveform']

        if available_parameters is not None:
            self.available_parameters = available_parameters
        else:
            self.available_parameters = list()

        if editable_parameters is not None:
            self.editable_parameters = editable_parameters
        else:
            self.editable_parameters = list()

    def _generate_wave(self, t):
        phase = self.phase.value % TWO_PI
        if self.input_module is not None:
            fm_modulation = self.input_module(t) * self.modulation_factor.value
        else:
            fm_modulation = torch.zeros_like(t)

        wave = torch.zeros_like(t)
        # Create the wave with fm modulation if given.
        if self.waveform.value == 'sine':
            wave = self.amp.value * torch.sin(TWO_PI * self.freq.value * t
                                              + phase + fm_modulation)
        elif self.waveform.value == 'square':
            wave = self.amp.value * torch.sign(torch.sin(TWO_PI * self.freq.value * t
                                                         + phase + fm_modulation))
        elif self.waveform.value == 'triangle':
            wave = (2 * self.amp.value / PI) * torch.arcsin(torch.sin((TWO_PI * self.freq.value * t
                                                                       + phase + fm_modulation)))
        elif self.waveform.value == 'sawtooth':
            # sawtooth closed form
            wave = 2 * (t * self.freq.value - torch.floor(0.5 + t * self.freq.value))

            # Phase shift by normalization to range [0,1] and modulo operation
            wave = (wave + 1) / 2 + (phase + fm_modulation) / TWO_PI

            # re-normalization to range [-amp, amp]
            wave = self.amp.value * (wave * 2 - 1)

        return wave

    @property
    def amp(self):
        return self._amp

    @property
    def freq(self):
        return self._freq

    @property
    def waveform(self):
        return self._waveform

    @property
    def phase(self):
        return self._phase

    def __str__(self):
        super_text = super(FMOscillator, self).__str__()
        amp_text = f"Amplitude: {self.amp.value}"
        freq_text = f"Frequency: {self.freq.value}"
        waveform_text = f"Waveform: {self.waveform.value}"
        phase_text = f"Phase: {self.phase.value}"
        modulation_text = f"Modulation: {self.modulation_factor.value}"
        return f"{super_text}\n  {amp_text}\n  {freq_text}\n  {waveform_text}\n  {phase_text}\n  {modulation_text}"


class ADSR(SynthModule):
    def __init__(self, input_module, output_module, is_active=True, device='cpu', name='', sample_rate=44100,
                 attack=0, decay=0, sustain=0, sustain_time=0, release=0,
                 available_parameters=[], editable_parameters=[]):
        super(ADSR, self).__init__(input_module, output_module, is_active, device, name, sample_rate)
        self._attack = ModuleParameter(value=attack, min_val=0.001, max_val=1)
        self._decay = ModuleParameter(value=decay, min_val=0.001, max_val=1)
        self._sustain = ModuleParameter(value=sustain, min_val=0.001, max_val=1)
        self._sustain_time = ModuleParameter(value=sustain_time, min_val=0.001, max_val=1)
        self._release = ModuleParameter(value=release, min_val=0.001, max_val=1)
        self.parameters = ['attack', 'decay', 'sustain', 'release', 'sustain_time']

        if available_parameters is not None:
            self.available_parameters = available_parameters
        else:
            self.available_parameters = list()

        if editable_parameters is not None:
            self.editable_parameters = editable_parameters
        else:
            self.editable_parameters = list()

    def _build_envelope(self, t):
        """
        Build ADSR envelope
        Variable note_off_time - sustain time is passed as parameter
        params:
            self: Self object with ['attack_t', 'decay_t', 'sustain_t', 'release_t', 'sustain_level'] parameters
        Returns:
            A torch with the constructed FM signal
        Raises:
            ValueError: Provided variables are out of range
        """
        input_time = t.shape[0]

        time_stamps = dict()
        time_stamps['attack_finish_time'] = min(input_time, self.attack.value * self.sample_rate)
        time_stamps['decay_finish_time'] = min(input_time, time_stamps['attack_finish_time']
                                               + self.decay.value * self.sample_rate)
        time_stamps['sustain_finish_time'] = min(input_time, time_stamps['decay_finish_time']
                                                 + self.sustain_time.value * self.sample_rate)
        time_stamps['release_finish_time'] = min(input_time, time_stamps['sustain_finish_time']
                                                 + self.release.value * self.sample_rate)

        attack_multiplier = torch.linspace(0, 1.0, int(self.attack.value * self.sample_rate), device=self.device)
        decay_multiplier = torch.linspace(1.0, self.sustain.value, int(self.decay.value * self.sample_rate), device=self.device)
        sustain_multiplier = self.sustain.value * torch.ones(int(self.sustain_time.value * self.sample_rate), device=self.device)
        release_multiplier = torch.linspace(self.sustain.value, 0, int(self.release.value * self.sample_rate), device=self.device)
        end_multiplier = torch.zeros(input_time, device=self.device)

        envelope = torch.concat([attack_multiplier,
                                 decay_multiplier,
                                 sustain_multiplier,
                                 release_multiplier,
                                 end_multiplier])[:input_time]

        return envelope

    def _generate_wave(self, t):
        if self.input_module is not None:
            envelope = self._build_envelope(t)
            enveloped_signal = self.input_module(t) * envelope
        else:
            enveloped_signal = torch.zeros_like(t)
        return enveloped_signal

    @property
    def attack(self):
        return self._attack

    @property
    def decay(self):
        return self._decay

    @property
    def sustain(self):
        return self._sustain

    @property
    def sustain_time(self):
        return self._sustain_time

    @property
    def release(self):
        return self._release

    def __str__(self):
        super_text = super(ADSR, self).__str__()
        attack_text = f"Attack: {self.attack.value}"
        decay_text = f"Decay: {self.decay.value}"
        sustain_text = f"Sustain: {self.sustain.value}"
        sustain_time_text = f"Sustain time: {self.sustain_time.value}"
        release_text = f"Release: {self.release.value}"
        return f"{super_text}\n  {attack_text}\n  {decay_text}\n  {sustain_text}\n  {sustain_time_text}\n" \
               f"  {release_text}"


class Filter(SynthModule):
    def __init__(self, input_module, output_module, is_active=True, device='cpu', name='', sample_rate=44100, modulator=None,
                 cutoff_freq=20000.0, resonance=0.1, filter_type='lowpass',
                 available_parameters=['cutoff_freq'], editable_parameters=['cutoff_freq'],
                 available_filter_types=['lowpass']):
        super(Filter, self).__init__(input_module, output_module, is_active, device, name, sample_rate, modulator)
        self._cutoff_freq = ModuleParameter(value=cutoff_freq, min_val=20, max_val=20000)
        self._resonance = ModuleParameter(value=resonance, min_val=0.1, max_val=100)
        self._filter_type = ModuleParameter(value=filter_type, options=available_filter_types)
        self.parameters = ['cutoff_freq', 'resonance', 'filter_type']

        if available_parameters is not None:
            self.available_parameters = available_parameters
        else:
            self.available_parameters = list()

        if editable_parameters is not None:
            self.editable_parameters = editable_parameters
        else:
            self.editable_parameters = list()

    def _generate_wave(self, t):
        if self.input_module is not None:
            if self.filter_type.value == 'lowpass':
                # wave = lowpass_biquad(self.input_module(t), self.sample_rate, self.cutoff_freq.value,
                #                       self.resonance.value).to(self.device)
                wave = lowpass_biquad(self.input_module(t).cpu(), self.sample_rate, self.cutoff_freq.value,
                                      self.resonance.value).to(self.device)
            elif self.filter_type.value == 'highpass':
                wave = highpass_biquad(self.input_module(t), self.sample_rate, self.cutoff_freq.value,
                                       self.resonance.value).to(self.device)
            elif self.filter_type.value == 'bandpass':
                wave = band_biquad(self.input_module(t), self.sample_rate, self.cutoff_freq.value,
                                   self.resonance.value).to(self.device)
        else:
            wave = torch.zeros_like(t)
        return wave

    @property
    def cutoff_freq(self):
        return self._cutoff_freq

    @property
    def resonance(self):
        return self._resonance

    @property
    def filter_type(self):
        return self._filter_type

    def __str__(self):
        super_text = super(Filter, self).__str__()
        cutoff_text = f"Cutoff frequency: {self.cutoff_freq.value}"
        resonance_text = f"Resonance factor: {self.resonance.value}"
        filter_type_text = f"Filter type: {self.filter_type.value}"
        return f"{super_text}\n  {cutoff_text}\n  {resonance_text}\n  {filter_type_text}"


class Mixer(SynthModule):
    def __init__(self, input_modules, output_module, is_active, device, name, sample_rate=44100):
        super(Mixer, self).__init__(None, output_module, is_active, device, name, sample_rate)
        self.input_modules = input_modules

    def _generate_wave(self, t):
        wave = torch.zeros_like(t)
        n_inputs = len(self.input_modules)
        for module in self.input_modules:
            wave = wave + module(t)
        return wave / n_inputs

    def connect_input(self, module: SynthModule):
        self.input_modules.append(module)
        module.output_module = self

    def disconnect_input(self, module: SynthModule):
        self.input_modules.remove(module)

    def disconnect_inputs(self):
        for module in self.input_modules:
            module.output_module = None
        self.input_modules = list()

    def __str__(self):
        super_text = super(Mixer, self).__str__()
        input_modules_text = '  Input modules: ['
        for module in self.input_modules:
            input_modules_text += module.name + ', '
        input_modules_text = input_modules_text[:-2] + ']'
        return super_text + '\n' + input_modules_text

    def __call__(self, t):
        if self.input_modules:
            self.signal = self._generate_wave(t)
            return self.signal
        else:
            return torch.zeros_like(t)


class Amplifier(SynthModule):
    def __init__(self, input_module, output_module, is_active=True, device='cpu', name='', sample_rate=44100,
                 amp_factor=1.0, modulator=None):
        super(Amplifier, self).__init__(input_module, output_module, is_active, device, name, sample_rate, modulator)
        self.amp_factor = amp_factor

    def _generate_wave(self, t):
        amp_factor = self.amp_factor * torch.ones_like(t)

        if self.modulator is not None:
            amp_factor = self.input_module(t) * self.modulator(t)

        if self.input_module is not None:
            return self.input_module(t) * amp_factor

        return torch.ones_like(t)


class Output(SynthModule):
    def _generate_wave(self, t):
        if self.input_module is not None:
            epsilon_noise = 0.0001 * torch.ones_like(t)
            return self.input_module(t) + epsilon_noise
        else:
            return torch.zeros_like(t)


def build_synth(config: dict, is_init=False, device='cpu'):
    assert 'n_oscillators' in config, "Must pick the number of oscillators in synth"

    n_oscillators = config['n_oscillators']
    sample_rate = config['sample_rate']
    signal_duration = config['signal_duration']
    available_waves = config['available_waves']
    available_filters = config['available_filters']

    available_parameters = dict()
    editable_parameters = dict()
    available_parameters_t = ['oscillator_available_parameters', 'filter_available_parameters', 'ADSR_available_parameters']
    editable_parameters_t = ['oscillator_editable_parameters', 'filter_editable_parameters', 'ADSR_editable_parameters']
    for (available_parameter, editable_parameter) in zip(available_parameters_t, editable_parameters_t):
        if available_parameter in config.keys():
            available_parameters[available_parameter] = config[available_parameter]
        else:
            available_parameters[available_parameter] = None

        if editable_parameter in config.keys():
            editable_parameters[editable_parameter] = config[editable_parameter]
        else:
            editable_parameters[editable_parameter] = None

    synth = Synthesizer(device=device, signal_duration=signal_duration, sample_rate=sample_rate,
                        n_oscillators=n_oscillators,
                        available_module_types=config["available_types"], editable_module_types=config["editable_types"],
                        available_waves=available_waves, available_filters=available_filters,
                        available_parameters=available_parameters, editable_parameters=editable_parameters)

    for key, value in config.items():
        current_module_key = key
        if type(value) == dict:
            input_module_key = value['input_module'] if 'input_module' in value else None
            output_module_key = value['output_module'] if 'output_module' in value else None

            current_module = synth.get_module_from_synth(current_module_key)
            input_module = synth.get_module_from_synth(input_module_key)
            output_module = synth.get_module_from_synth(output_module_key)

            for attr, attr_val in value.items():
                if attr in ['input_module', 'output_module']:
                    continue
                attr_object = current_module.__getattribute__(attr)
                if is_init:
                    attr_object.value = 0.0001
                else:
                    attr_object.value = attr_val

            if input_module is not None:
                synth.connect_modules(input_module, current_module)  # Connect input module to current as input

            if output_module is not None:
                synth.connect_modules(current_module, output_module)  # Connect current to output as input
    return synth
