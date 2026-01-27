import numpy as np
import torch
from numpy import void
from torch import nn
from typing import Any, cast, Dict, List, Optional, Union, Callable, Type
import math

eps = 0.00001
stam = False


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=2)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        identity = self.downsample(x)
        x = self.conv1(x)
        # x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.batch2(x)

        x = x + identity
        x = self.relu(x)
        return x


class BackBone(nn.Module):
    def __init__(
            self,
            layers: List[int],
            representation_dim: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            sample_rate: int = 44100
    ) -> None:
        super(BackBone, self).__init__()

        self.inplanes = 1
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], kernel_size=(5, 5))
        self.layer2 = self._make_layer(96, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(144, layers[2], stride=1, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(216, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(216 * ResBlock.expansion, representation_dim)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.batch2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            kernel_size: tuple = (5, 5)
    ) -> nn.Sequential:

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * ResBlock.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * ResBlock.expansion, (1, 1), stride=1)

        layers = list()
        layers.append(
            ResBlock(in_channels=self.inplanes,
                     out_channels=planes,
                     kernel_size=kernel_size,
                     stride=stride)
        )
        self.inplanes = planes * ResBlock.expansion

        for _ in range(1, blocks):
            layers.append(
                ResBlock(
                    self.inplanes,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, output_dim: int, n_layers: int,
                 mid_activation, input_activation='Tanh', output_activation='LeakyReLU',
                 norm: bool = False):
        super(MLP, self).__init__()
        # First layer initialization
        _, input_std = get_activation_and_std(input_activation, input_dim)

        # Intermediate layers initialization
        self.mid_activation, mid_std = get_activation_and_std(mid_activation, model_dim)

        # Output layer
        self.out_activation, out_std = get_activation_and_std(output_activation, model_dim)

        self.layers = nn.ModuleList([nn.Linear(input_dim, model_dim)])  # List of layers
        nn.init.normal_(self.layers[0].weight, 0, input_std)

        # Extending the list of layers with a list of layers
        self.layers.extend([nn.Linear(model_dim, model_dim) for i in range(n_layers - 1)])
        for layer in self.layers[1:]:
            nn.init.normal_(layer.weight, 0, mid_std)

        self.output = nn.Linear(model_dim, output_dim)
        nn.init.normal_(self.output.weight, 0, out_std)

        self.norm = norm
        if norm:
            self.norms = nn.ModuleList([])
            self.norms.extend([RMSNorm(model_dim) for i in range(n_layers)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norm:
                x = self.norms[i](x)
            x = self.mid_activation(x)
            if stam:
                print(i)
                print(x)
        x = self.output(x)
        x = self.out_activation(x)
        if stam:
            print(i + 1)
            print(x)
        return x


class RLMLP(nn.Module):
    def __init__(self, representation_dim, n_representations, context_dim, model_dim, n_layers, action_dim=0,
                 is_residual=False, activation='Tanh', is_deep_wide=False):
        super(RLMLP, self).__init__()
        self.representation_dim = representation_dim
        self.n_representations = n_representations
        self.signals_representation_dim = n_representations * representation_dim
        self.is_residual = is_residual
        self.is_deep_wide = is_deep_wide
        input_to_mlp_dim = self.signals_representation_dim

        self.context_embedding = None
        if context_dim > 0:
            if self.is_deep_wide:
                self.context_embedding = MLP(input_dim=context_dim, model_dim=model_dim, output_dim=representation_dim,
                                             n_layers=2,
                                             input_activation='Tanh',
                                             mid_activation='Tanh',
                                             output_activation='Tanh')
                input_to_mlp_dim += representation_dim

        self.action_embedding = None
        if action_dim > 0:
            if self.is_deep_wide:
                self.action_embedding = MLP(input_dim=action_dim, model_dim=model_dim, output_dim=representation_dim,
                                            n_layers=2,
                                            input_activation='Tanh',
                                            mid_activation='LeakyRelu',
                                            output_activation='Tanh')
                input_to_mlp_dim += representation_dim

        if is_residual:
            input_to_mlp_dim += action_dim
            input_to_mlp_dim += context_dim

        self.mlp = MLP(input_dim=input_to_mlp_dim, model_dim=model_dim, output_dim=model_dim, n_layers=n_layers,
                       input_activation='Tanh', mid_activation=activation, output_activation='Tanh')

    def forward(self, signal_representations, context=None, action=None):
        tensor_list = signal_representations.copy()
        if self.is_residual:
            if context is not None:
                tensor_list.append(context)
            if action is not None:
                tensor_list.append(action)

        if self.is_deep_wide and context is not None:
            context_embedding = self.context_embedding(context)
            tensor_list.append(context_embedding)

        if self.is_deep_wide and action is not None:
            action_embedding = self.action_embedding(action)
            tensor_list.append(action_embedding)
        input2model = torch.concat(tensor_list)
        h = self.mlp(input2model)
        return h


class SimpleCNN(nn.Module):
    def __init__(self, n_filters: List[int], kernel_sizes: List[int], representation_dim: int, n_frames, n_mels=128,
                 n_sounds=1):
        super(SimpleCNN, self).__init__()
        alpha = 0.1
        self.activation = nn.LeakyReLU(negative_slope=alpha)
        self.activation = nn.Tanh()
        variance_multiplier = 2 / (1 + np.power(alpha, 2))
        input_std = np.sqrt(variance_multiplier / (n_filters[-1] * (n_frames + n_mels)))

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=n_sounds, out_channels=n_filters[0],
                                              kernel_size=kernel_sizes[0],
                                              padding=int((kernel_sizes[0] - 1) / 2))])  # List of layers

        self.downsamples = nn.ModuleList([nn.Conv2d(in_channels=n_sounds, out_channels=n_filters[0], kernel_size=1)])
        for i, (n, k) in enumerate(zip(n_filters[1:], kernel_sizes[1:])):
            padding = int((k - 1) / 2)
            self.convs.extend([nn.Conv2d(in_channels=n_filters[i],
                                         out_channels=n,
                                         kernel_size=k,
                                         padding=padding)])

            self.downsamples.extend([nn.Conv2d(in_channels=n_filters[i],
                                               out_channels=n,
                                               kernel_size=1)])

        self.frequency_summarization = nn.Conv2d(in_channels=n_filters[-1],
                                                 out_channels=n_filters[-1],
                                                 # groups=n_filters[-1],
                                                 kernel_size=(n_mels, 1))

        self.time_summarization = nn.Conv2d(in_channels=n_filters[-1],
                                            out_channels=n_filters[-1],
                                            # groups=n_filters[-1],
                                            kernel_size=(1, n_frames))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, a=alpha, nonlinearity="leaky_relu")
                nn.init.xavier_normal_(m.weight)

        after_filters_dim = n_filters[-1] * (n_frames * n_mels)
        # after_filters_dim = n_filters[-1] * n_frames
        self.features2encoding = MLP(after_filters_dim, representation_dim, representation_dim, 1,
                                     input_activation='Tanh',
                                     mid_activation='Tanh',
                                     output_activation='Tanh')

        # self.features2encoding = nn.Linear(n_filters[-1] * (n_frames * n_mels), representation_dim)
        # nn.init.xavier_normal_(self.features2encoding.weight)

    def forward(self, x):
        global stam

        for conv, downsample in zip(self.convs, self.downsamples):
            x = conv(x)
            x = self.activation(x)

        # frequency_summarization = self.frequency_summarization(x).squeeze(-1)
        # frequency_summarization = torch.flatten(frequency_summarization)
        # x = frequency_summarization

        #
        # time_summarization = self.time_summarization(x).squeeze(-1)
        # time_summarization = torch.flatten(time_summarization)
        #
        # x = torch.concat([frequency_summarization, time_summarization])
        x = torch.flatten(x)
        x = self.activation(x)
        # stam = True
        x = self.features2encoding(x)
        stam = False
        return x


class SimpleCNNDecoder(nn.Module):
    def __init__(self, n_filters: List[int], kernel_sizes: List[int], representation_dim: int, n_frames):
        super(SimpleCNNDecoder, self).__init__()
        alpha = 0.1
        self.n_frames = n_frames
        self.n_filters = n_filters
        self.convs = nn.ModuleList([nn.ConvTranspose2d(in_channels=1,
                                                       out_channels=n_filters[0],
                                                       kernel_size=(n_filters[0] + 1, 1),
                                                       stride=1)])  # List of layers

        for i, (n, k) in enumerate(zip(n_filters[1:], kernel_sizes[1:])):
            self.convs.extend([nn.ConvTranspose2d(in_channels=n_filters[i],
                                                  out_channels=n,
                                                  kernel_size=k,
                                                  padding=1)])

        self.final_conv = nn.Conv2d(in_channels=n_filters[-1],
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, a=alpha, nonlinearity="leaky_relu")

        self.activation = nn.LeakyReLU(negative_slope=alpha)

        self.encoding2features = nn.Linear(representation_dim, n_filters[0] * n_frames)
        variance_multiplier = 2 / (1 + np.power(alpha, 2))
        nn.init.normal_(self.encoding2features.weight, 0, variance_multiplier / np.sqrt(n_filters[0] * n_frames))

    def forward(self, x):
        x = self.encoding2features(x)
        x = x.view((-1, self.n_filters[0], self.n_frames))

        for conv in self.convs:
            # identity = downsample(x)
            x = conv(x)
            x = self.activation(x)
            # x = x + identity
            # x = self.maxpool(x)

        x = self.final_conv(x)

        return x


class EncoderDecoder(nn.Module):
    def __init__(self, n_filters: List[int], kernel_sizes: List[int], representation_dim: int, n_frames):
        super(EncoderDecoder, self).__init__()
        self.encoder = SimpleCNN(n_filters,
                                 kernel_sizes,
                                 representation_dim,
                                 n_frames)
        self.decoder = SimpleCNNDecoder(list(reversed(n_filters)),
                                        list(reversed(kernel_sizes)),
                                        representation_dim,
                                        n_frames)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SimpleCNN1D(nn.Module):
    def __init__(self, n_filters, kernel_sizes, representation_dim):
        super(SimpleCNN1D, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters[0],
                                              kernel_size=kernel_sizes[0],
                                              padding=int((kernel_sizes[0] - 1) / 2),
                                              stride=128)])  # List of layers

        self.downsamples = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=n_filters[0], kernel_size=1)])
        for i, (n, k) in enumerate(zip(n_filters[1:], kernel_sizes[1:])):
            padding = int((k - 1) / 2)
            self.convs.extend([nn.Conv1d(in_channels=n_filters[i],
                                         out_channels=n,
                                         kernel_size=k,
                                         padding=padding,
                                         stride=128)])

            self.downsamples.extend([nn.Conv1d(in_channels=n_filters[i], out_channels=n, kernel_size=1)])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        self.activation = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(n_filters[-1], representation_dim)
        nn.init.normal_(self.fc.weight, 0, 2 / np.sqrt(representation_dim))

    def forward(self, x):
        for conv, downsample in zip(self.convs, self.downsamples):
            # identity = downsample(x)
            x = conv(x)
            x = self.activation(x)
            # x = x + identity
            # x = self.maxpool(x)

        x = self.avgpool(x).squeeze(-1)
        x = torch.flatten(x)
        x = self.fc(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, n_filters_spec, kernel_sizes_spec, n_filters_wave, kernel_sizes_wave,
                 representation_dim, model_dim, n_layers):
        super(FusionModel, self).__init__()
        representation_std = np.sqrt(2 / (2 * representation_dim))

        self.melspec_features_model = SimpleCNN(n_filters_spec, kernel_sizes_spec, representation_dim)
        self.waveform_features_model = SimpleCNN1D(n_filters_wave, kernel_sizes_wave, representation_dim)
        self.fc = nn.Linear(2 * representation_dim, representation_dim)  # List of layers
        nn.init.normal_(self.fc.weight, 0, representation_std)

        self.activation = nn.ReLU()

    def forward(self, waveform, melspec):
        melspec_features = self.melspec_features_model(melspec)
        waveform_features = self.waveform_features_model(waveform)
        x = torch.concat([melspec_features, waveform_features])
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        if d_model % 2 != 0:
            n_odd = int((d_model + 1) / 2)
            n_even = int(d_model / 2)
            position_odd = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_odd).t()
            position_even = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_even).t()
        else:
            n_odd = n_even = int(d_model / 2)
            position_odd = position_even = torch.arange(max_len).unsqueeze(0).t().repeat(1, n_even).t()
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).unsqueeze(1).repeat(1, max_len)
        pe = torch.zeros(1, d_model, max_len)
        pe[0, 0::2] = torch.sin(position_odd * div_term[:n_odd])
        pe[0, 1::2] = torch.cos(position_even * div_term[:n_even])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return x


class AttentionModel(nn.Module):
    def __init__(self, n_mels, n_frames, model_dim, representation_dim):
        super(AttentionModel, self).__init__()
        self.freq_pos_code = PositionalEncoding(d_model=n_mels,
                                                max_len=n_frames).to('cuda')

        self.time_pos_code = PositionalEncoding(d_model=n_frames,
                                                max_len=n_mels).to('cuda')

        self.freq_projection = nn.Linear(n_frames, model_dim)
        self.time_projection = nn.Linear(n_mels, model_dim)
        nn.init.normal_(self.freq_projection.weight, 0, np.sqrt(1 / n_frames))
        nn.init.normal_(self.time_projection.weight, 0, np.sqrt(1 / n_mels))

        self.attention = nn.MultiheadAttention(model_dim, 4)

        self.attention2representation = nn.Linear((n_mels + n_frames) * model_dim, representation_dim)
        nn.init.normal_(self.attention2representation.weight, 0, np.sqrt(1 / ((n_mels + n_frames) * model_dim)))

        self.tanh = nn.Tanh()

    def forward(self, mel_spec):
        freq_encoded = self.freq_pos_code(mel_spec)
        time_encoded = self.time_pos_code(mel_spec.transpose(1, 2))

        freq_projection = self.freq_projection(freq_encoded)
        time_projection = self.time_projection(time_encoded)

        freq, freq_mask = self.attention(freq_projection, freq_projection, freq_projection)
        time, time_mask = self.attention(time_projection, time_projection, time_projection)

        attention = torch.concat([freq.flatten(), time.flatten()])
        representation = self.attention2representation(attention)
        representation = self.tanh(representation)
        return representation


class CategoricalChoice(nn.Module):
    def __init__(self, representation_dim, n_representations, context_dim, choices: list, model_dim, n_layers,
                 is_residual=False, are_shared_mlp=False, is_deep_wide=False):
        super(CategoricalChoice, self).__init__()
        self.choices = choices
        self.n_choices = len(choices)
        self.are_shared_mlp = are_shared_mlp

        # If the models share a mlp and differ by heads we don't need multiple representations as input, thus skipping
        # the mlp
        if not are_shared_mlp:
            self.mlp = RLMLP(representation_dim, n_representations, context_dim, model_dim, n_layers,
                             is_residual=is_residual, is_deep_wide=is_deep_wide)

        self.choice = nn.Linear(model_dim, self.n_choices)
        self.termination = nn.Linear(model_dim, 1)

        scale = 100
        nn.init.normal_(self.choice.weight, 0, 2 / (scale * np.sqrt(model_dim)))
        nn.init.normal_(self.termination.weight, 0, 2 / (scale * np.sqrt(model_dim)))

        nn.init.zeros_(self.choice.bias)
        nn.init.zeros_(self.termination.bias)

    def forward(self, signal_representations=None, context=None, advanced_representation=None):
        if self.are_shared_mlp:
            h = advanced_representation
        else:
            h = self.mlp(signal_representations, context)

        logits = self.choice(h)

        termination = self.termination(h)
        termination = torch.sigmoid(termination)
        return logits, termination

    def sample_based_on_state(self, probabilities, is_optimal=False, device='cpu'):
        if is_optimal:
            result = torch.argmax(probabilities).unsqueeze(0)
        else:
            result = sample_categorical(probabilities)
        return result


class ParameterDifferenceRegression(nn.Module):
    def __init__(self, representation_dim, n_representations, context_dim, max_step_val, model_dim, n_layers,
                 is_residual=False, are_shared_mlp=False, is_deep_wide=False):
        super(ParameterDifferenceRegression, self).__init__()
        self.are_shared_mlp = are_shared_mlp
        if not are_shared_mlp:
            self.mlp = RLMLP(representation_dim, n_representations, context_dim, model_dim, n_layers,
                             is_residual=is_residual, is_deep_wide=is_deep_wide)

        self.max_step_val = max_step_val
        self.mu = nn.Linear(model_dim, 1)
        self.sigma = nn.Linear(model_dim, 1)
        self.termination = nn.Linear(model_dim, 1)
        self.softplus = nn.Softplus()

        scale = 100
        nn.init.normal_(self.mu.weight, 0, 1 / (scale * np.sqrt(model_dim)))
        nn.init.normal_(self.sigma.weight, 0, 1 / (scale * np.sqrt(model_dim)))
        nn.init.normal_(self.termination.weight, 0, 1 / (scale * np.sqrt(model_dim)))

        nn.init.zeros_(self.mu.bias)
        nn.init.constant_(self.sigma.bias, -1)
        nn.init.zeros_(self.termination.bias)

    def forward(self, signal_representations=None, context=None, advanced_representation=None):
        if self.are_shared_mlp:
            h = advanced_representation
        else:
            h = self.mlp(signal_representations, context)

        mu = self.mu(h)
        step = torch.tanh(mu)
        extremity = torch.abs(step).detach()
        mu = step * self.max_step_val

        temp_sigma = self.sigma(h)
        # sigma = torch.sigmoid(temp_sigma) * (self.max_step_val / 3)
        sigma = self.softplus(temp_sigma) * (self.max_step_val / 3)
        # sigma = torch.sigmoid(temp_sigma) * abs(mu.item()) * 3

        termination = self.termination(h)
        termination = torch.sigmoid(termination)

        return [mu, sigma, extremity], termination


class ValueEstimation(nn.Module):
    def __init__(self, representation_dim, n_representations, context_dim, model_dim, n_layers, is_residual=False,
                 are_shared_mlp=False, is_deep_wide=False, scale_factor=1):
        super(ValueEstimation, self).__init__()
        self.are_shared_mlp = are_shared_mlp
        dim_for_value = model_dim
        if not are_shared_mlp:
            dim_for_value = model_dim
            self.mlp = RLMLP(representation_dim, n_representations, context_dim, model_dim, n_layers,
                             is_residual=is_residual, is_deep_wide=is_deep_wide)

        self.scale_factor = scale_factor / 2
        self.value = nn.Linear(dim_for_value, 1)
        self.multiplier = torch.nn.Parameter(torch.ones(1))
        self.tanh = nn.Tanh()

        scale = 10
        nn.init.normal_(self.value.weight, 0, 1 / (scale * np.sqrt(model_dim)))
        nn.init.normal_(self.value.bias, 0, 1 / np.sqrt(model_dim))

    def forward(self, signal_representations=None, context=None, advanced_representation=None):
        if self.are_shared_mlp:
            h = advanced_representation
        else:
            h = self.mlp(signal_representations, context)
        # h = self.value(h * self.multiplier)
        h = self.value(h) * self.scale_factor
        return h


class ParameterRegression(nn.Module):
    def __init__(self, representation_dim, n_parameters, model_dim, n_layers):
        super(ParameterRegression, self).__init__()
        self.n_parameters = n_parameters
        self.mlp = MLP(input_dim=representation_dim, model_dim=model_dim, output_dim=model_dim, n_layers=n_layers,
                       mid_activation='LeakyReLU')

        self.mu = nn.Linear(model_dim, self.n_parameters)
        nn.init.normal_(self.mu.weight, 0, 1 / np.sqrt(n_parameters))

    def forward(self, x):
        x = self.mlp(x)
        mu = self.mu(x)
        mu = torch.sigmoid(mu)

        return mu

    def sample_based_on_state(self, x, is_optimal=False):
        mu, sigma, extremity = self(x)
        if is_optimal:
            return mu, 0
        else:
            value = sample_continuous(mu=mu, sigma=sigma)
        return value, extremity


class StateRegression(nn.Module):
    def __init__(self, representation_dim, model_dim, action_dim, context_dim, n_layers, is_residual=True):
        super(StateRegression, self).__init__()
        self.mlp = RLMLP(representation_dim, 1, context_dim, model_dim, n_layers, action_dim, is_residual=is_residual)

        self.state = nn.Linear(model_dim, representation_dim)
        nn.init.normal_(self.state.weight, 0, 2 / np.sqrt(model_dim))
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal_representations, action, context=None):
        x = self.mlp(signal_representations, context, action)
        x = self.state(x)
        x = torch.tanh(x)
        return x


class ActionRegression(nn.Module):
    def __init__(self, representation_dim, model_dim, action_dim, context_dim, n_layers, is_residual=True):
        super(ActionRegression, self).__init__()
        self.action_dim = action_dim
        self.mlp = RLMLP(representation_dim, 2, context_dim, model_dim, n_layers)

        self.action = nn.Linear(model_dim, action_dim)
        nn.init.normal_(self.action.weight, 0, 1 / np.sqrt(model_dim))

    def forward(self, signal1_representation, signal2_representation, context=None):
        signals_representations = [signal1_representation, signal2_representation]
        h = self.mlp(signals_representations, context)
        x = self.action(h)
        if self.action_dim == 1:
            x = torch.tanh(x)
        return x


class ParametersEncoder(nn.Module):
    def __init__(self, n_parameters, model_dim, representation_dim, n_layers):
        super(ParametersEncoder, self).__init__()
        self.mlp = MLP(input_dim=n_parameters, model_dim=model_dim, output_dim=model_dim, n_layers=n_layers,
                       mid_activation='LeakyReLU')

        self.encoding = nn.Linear(model_dim, representation_dim)
        nn.init.normal_(self.encoding.weight, 0, 1 / np.sqrt(model_dim))

    def forward(self, x):
        x = self.mlp(x)
        x = self.encoding(x)
        x = torch.tanh(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def sample_continuous(mu, sigma):
    global eps
    dist = torch.distributions.Normal(mu, sigma + eps)

    x = dist.sample()
    return x


def get_log_prob_continuous(x, mu, sigma):
    dist = torch.distributions.Normal(mu, sigma + eps)
    x_tensor = x.clone()
    log_prob = dist.log_prob(x_tensor)

    return log_prob


def sample_categorical(probabilities):
    result = probabilities.multinomial(num_samples=1)
    return result


def get_log_prob_categorical(x, probabilities):
    log_prob = torch.log(probabilities[x])[0]
    return log_prob


def expend_layer(layer: nn.Module, new_output_size):
    prev_weight = layer.weight.data
    prev_bias = layer.bias.data
    prev_weight_shape = prev_weight.shape
    prev_bias_shape = prev_bias.shape

    new_layer = nn.Linear(prev_weight_shape[0], new_output_size)

    new_layer.weight.data[:prev_weight_shape[0], :prev_weight_shape[1]] = prev_weight
    new_layer.bias.data[:prev_bias_shape[0]] = prev_bias

    return new_layer


def get_activation_and_std(activation, dim):
    if activation == 'LeakyRelu':
        alpha = 0.1
        variance_multiplier = 2 / (1 + np.power(alpha, 2))
        std = np.sqrt(variance_multiplier / dim)
        activation_module = nn.LeakyReLU(negative_slope=alpha)

    elif activation == 'Tanh':
        std = np.sqrt(1 / dim)
        activation_module = nn.Tanh()

    elif activation == 'Linear':
        std = np.sqrt(1 / dim)
        activation_module = nn.Identity()

    else:
        std = np.sqrt(2 / dim)
        activation_module = nn.ReLU()

    return activation_module, std
