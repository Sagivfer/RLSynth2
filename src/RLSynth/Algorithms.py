import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Type, List
import torch.optim as optim
from _distutils_hack import shim
from torch.utils.data import BatchSampler

import src.RLSynth.Environment as Env
import src.RLSynth.Utils as UT
from src.RLSynth.prioritized_memory import Memory
import wandb

torch.set_printoptions(precision=10)


class RLOptimizer(object):
    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: [Env.HierarchicalAgent, Env.SingleActionAgent],
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict,
                 test_set: list,
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):

        self.optimizer_agent = optimizer_agent
        self.optimizer_critic = optimizer_critic
        self.lr_scheduler_agent = lr_scheduler_agent
        self.lr_scheduler_critic = lr_scheduler_critic

        self.gamma = config['gamma']
        self.etta = config['etta']
        self.env = env
        self.agent = agent
        self.critic = critic
        self.n_steps = n_steps
        self.n_steps_for_update = 0
        # self.n_episodes = config['n_episodes']
        self.k_epochs = config['k_epochs']
        self.epsilon = config['epsilon']
        self.T = T
        self.n_modules = n_modules
        self.n_parameters = n_parameters
        self.R = list()
        self.batch_size = config['batch_size']

        self.is_with_baseline = config['baseline']
        self.device = config['device']
        self.n_samples = config['n_samples']

        self.losses = list()
        self.loss = torch.tensor(0)

        self.batch_count = 0

        self.SRL = config['SRL']
        self.SRL_weight = config['SRL_weight']
        self.alt = config['alt']
        self.alt_every = config['alt_every']
        self.count_for_alt = 0
        self.RL_weight = 1
        self.normalize_reward = config['normalize_reward']

        self.pretrained = is_pretrained
        self.are_shared_backbone = are_shared_backbone
        self.temperature_decrease_rate = config['temperature_decrease_rate']

        self.config = config

        what2log = ['intermediate_reward', 'advantage', 'critic_error', 'td_target',
                    'explained_variance', 'final_reward', 't', 'critic_value',
                    'immediate_reward', 'exploration', 'final_reward_policy',
                    'param_loss', 'state_regression_loss', 'action_regression_loss',
                    'reward_time_1', 'reward_time_2', 'reward_time_3', 'Single_Agent_advantage',
                    'advantage_cut_off', 'G0']

        self.level2agent = Env.DFS_agent(self.agent.top_agent, level2agent=dict())

        if logger is None:
            self.logger = UT.Logger(config['average_type'], 0.99)
            self.advantage_logger = UT.Logger(config['average_type'], 0.33)

            for log in what2log:
                self.logger.set_up_log(log)

            for m in self.env.synth.modules:
                for parameter in m.available_parameters:
                    synth_parameter_object = m.__getattribute__(parameter)
                    log_name = f'{m.name}_{parameter}' if self.env.hierarchy_type == 'module' else f'{parameter}_{m.name}'
                    if len(synth_parameter_object.options) == 0:
                        self.logger.set_up_log(f'{log_name}_diff')

            # for t in range(T):
                # self.advantage_logger.set_up_log(f'{self.agent.module_name}_advantage_{t}')

                # for level in self.level2agent.keys():
                #     for agent in self.level2agent[level]:
                #         self.advantage_logger.set_up_log(f'{agent.module_name}_advantage_{t}')

            for level in self.level2agent.keys():
                for agent in self.level2agent[level]:
                    self.logger.set_up_log(f'{agent.module_name}')
                    self.logger.set_up_log(f'{agent.module_name}_extremity')
                    self.logger.set_up_log(f'{agent.module_name}_dwell')
                    self.advantage_logger.set_up_log(f'{agent.module_name}_advantage')
                    self.advantage_logger.set_up_log(f'{agent.module_name}_advantage_std')

        else:
            self.logger = logger
            self.advantage_logger = advantage_logger

        self.is_clipping = config['is_clipping']
        self.clipping = config['clipping']

        self.is_ET = config['is_ET']
        self.is_TD = config['is_TD']

        self.is_hard_search = config["is_hard_search"]

        self.connected2wifi = config["connected2wifi"]

        self.test_every = config["test_every"]
        self.count_test = 0
        self.test_set = test_set

        if self.connected2wifi:
            if self.config is not None and run is None:
                self.run = wandb.init(
                    project='Synth_with_RL',
                    config=self.config
                )
            else:
                self.run = run
                self.logger.step = self.run.step

    def optimize(self):
        # torch.autograd.set_detect_anomaly(True)
        self.agent.to(self.device)
        self.critic.to(self.device)

    def get_state_regression_loss(self, state_prediction, current_state_representation):
        loss = torch.mean(torch.pow(state_prediction - current_state_representation, 2))
        self.logger.update_average('state_regression_loss', loss.item())
        return loss

    def get_action_regression_loss(self, agent, action_prediction, action):
        if agent.selection is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            loss = ce_loss(action_prediction, action[0][3].detach())
        else:
            loss = torch.sum(torch.pow(action_prediction - action[0][2].detach() / agent.regression.max_step_val, 2))
        self.logger.update_average('action_regression_loss', loss.item())
        return loss

    def get_SRL_loss(self, agent, action, transition):
        parameter_loss = torch.zeros(1, device=self.device)
        state_regression_loss = torch.zeros(1, device=self.device)
        action_regression_loss = torch.zeros(1, device=self.device)

        if agent.state_regression is not None:
            with torch.no_grad():
                previous_representation = self.agent.get_sound_representation(transition.state1[1])
                current_representation = self.agent.get_sound_representation(transition.state2[1])
            signal_prediction, action_prediction = agent.get_SRL_predictions(action, transition.state1[2],
                                                                             previous_representation,
                                                                             current_representation)
            state_regression_loss = self.get_state_regression_loss(signal_prediction, current_representation)
            action_regression_loss = self.get_action_regression_loss(agent, action_prediction, action)
        return self.SRL_weight * (0 * parameter_loss + 0 * state_regression_loss + 0.005 * action_regression_loss)

    def get_actor_loss(self, advantage, action_log_prob, transition):
        # print(action_log_prob, advantage)
        loss = -1 * self.RL_weight * action_log_prob * advantage
        return loss

    def get_critic_loss(self, critic_error):
        if not self.is_with_baseline:
            return torch.zeros(1, device=self.device)

        self.logger.update_average('critic_error', critic_error.item())
        critic_loss = torch.zeros(1, device=self.device)
        if critic_error != 0:
            critic_loss = torch.pow(critic_error, 2)
        explained_variance_new = (self.logger.std['critic_error'] ** 2) / (self.logger.std['td_target'] ** 2)
        self.logger.update_average('explained_variance', 1 - explained_variance_new)
        return self.RL_weight * critic_loss

    def get_critic_lipschitz_reg(self, critic_td_error, agent, action):
        # return 0

        d_y = torch.abs(critic_td_error)
        d_x = 0.000001
        if agent.regression is None:
            return 0

        for sub_action in action:
            if isinstance(action, list):
                d_x = d_x + torch.abs(sub_action[-3] / agent.max_val)

        lipschitz_continuity = d_y / d_x
        # print(lipschitz_continuity, 'lipschitz')
        return 0.001 * lipschitz_continuity

    def get_extremity_reg(self, policy):
        if policy is None:
            return torch.zeros(1, device=self.device)

        extremity_loss = torch.zeros(1, device=self.device)
        for sub_policy in policy:
            if isinstance(sub_policy, list):
                extremity_loss = extremity_loss + sub_policy[2]

        # print(f"Extremity loss: {extremity_loss}")
        return 0.001 * self.RL_weight * extremity_loss

    def get_entropy_reg(self, policy):
        if policy is None:
            return torch.zeros(1, device=self.device)

        entropy_loss = torch.zeros(1, device=self.device)
        for sub_policy in policy:
            if isinstance(sub_policy, list):
                entropy_loss = entropy_loss + 0 * sub_policy[1]
            else:
                entropy_loss = entropy_loss - 0.1 * torch.sum(sub_policy * torch.log(sub_policy))
        # print(f"Entropy loss: {entropy_loss}")
        return -1 * self.RL_weight * entropy_loss

    def get_logits_reg(self, logits_list):
        logits_loss = torch.zeros(1, device=self.device)
        for logits in logits_list:
            if isinstance(logits, list):
                continue
            if logits.grad_fn is not None:
                logits_loss = torch.mean(torch.abs(logits))
        # print(f"Logits loss: {logits_loss}, {logits}")
        return 0 * self.RL_weight * logits_loss

    def get_stopping_loss(self, delta, stop_prob):
        stopping_loss = torch.zeros(1, device=self.device)
        stopping_regularization = torch.zeros(1, device=self.device)
        if stop_prob is not None:
            stopping_loss = (delta + self.etta) * stop_prob
            stopping_regularization = torch.abs(0.5 - stop_prob)

        # print(f"Stopping loss: {stopping_loss}, Stopping regularization: {stopping_regularization}")
        return self.RL_weight * (stopping_loss + 0 * stopping_regularization)

    def get_representation_reg(self, state_representation):
        state_representation_reg = torch.mean(torch.abs(state_representation))
        # print(f"Representation loss: {state_representation_reg}")
        return 0 * self.RL_weight * state_representation_reg

    def calc_loss_components(self, agent, action, advantage, action_log_prob, critic_error, critic_td_error,
                             policy, logits, transition,
                             stop_prob=None, signal_representation=None):
        loss_weight = self.get_loss_weight(agent)
        self.losses.append(self.get_critic_loss(critic_error) * loss_weight)
        self.losses.append(self.get_critic_lipschitz_reg(critic_td_error, agent, action) * loss_weight)

        # Updating the temporary average so all the advantages are equally weighted
        self.advantage_logger.update_batch_average(f'{agent.module_name}_advantage', advantage)
        advantage_for_stopping = advantage
        if self.normalize_reward and self.logger.loss_calc_step > 10000:
            std = self.advantage_logger.std[f'{agent.module_name}_advantage_std'] + 1e-8
            advantage = (advantage - self.advantage_logger.averages[f'{agent.module_name}_advantage']) / std

        self.losses.append(self.get_actor_loss(advantage, action_log_prob, transition) * loss_weight)
        self.losses.append(self.get_representation_reg(signal_representation) * loss_weight)
        self.losses.append(self.get_stopping_loss(advantage_for_stopping, stop_prob) * loss_weight)
        self.losses.append(self.get_extremity_reg(policy) * loss_weight)
        if len(agent.next_agents) > 0:
            self.losses.append(self.get_entropy_reg(policy) * loss_weight)
        self.losses.append(self.get_logits_reg(logits) * loss_weight)
        if self.SRL and not self.pretrained:
            self.losses.append(self.get_SRL_loss(agent, action, transition))
        loss_text = f'{agent.module_name}, '
        self.loss = self.loss + sum(self.losses)
        UT.delete_iterable_items(self.losses)

    def get_loss_weight(self, agent):
        loss_weight = 1
        module_name = agent.module_name
        if module_name == 'Filter0':
            loss_weight = 1
        elif module_name in ['Oscillator0', 'Oscillator1']:
            # loss_weight = 1 if self.logger.loss_calc_step > 500000 else 0
            loss_weight = 1

        elif module_name == 'TopAgent':
            # loss_weight = 1 if self.logger.loss_calc_step > 500000 else 0
            loss_weight = 1

        elif module_name == 'waveform':
            loss_weight = 1
        return loss_weight

    def update_parameters_wrt_grads(self, delta=1):
        # self.print_magnitude()
        if self.critic is not None:
            if self.is_clipping:
                for p in self.critic.model_parameters:
                    torch.nn.utils.clip_grad_norm_(p['params'], self.clipping)

            if isinstance(self.optimizer_critic, UT.AdamTrace):
                self.optimizer_critic.step(delta=delta)
            else:
                self.optimizer_critic.step()

            self.optimizer_critic.zero_grad()
            if self.are_shared_backbone:
                return

        if self.is_clipping:
            for p in self.agent.model_parameters:
                torch.nn.utils.clip_grad_norm_(p['params'], self.clipping)

        if isinstance(self.optimizer_agent, UT.AdamTrace):
            self.optimizer_agent.step(delta=delta)
        else:
            self.optimizer_agent.step()

        self.optimizer_agent.zero_grad()

    def update_learning_schedulers(self):
        self.lr_scheduler_agent.step()
        if self.lr_scheduler_critic is not None:
            self.lr_scheduler_critic.step()

    def update_grads(self, retain_graph: bool = False):
        self.loss = self.loss / self.batch_size

        self.batch_count = 0
        if self.loss.grad_fn is not None:
            if retain_graph:
                self.loss.backward(retain_graph=retain_graph)
            else:
                self.loss.backward()
                del self.loss
                self.loss = torch.tensor(0)

    def test_policy(self, test_size=10):
        final_reward = 0
        for i in range(test_size):
            self.env.randomize_synth(randomize_type=3)
            Env.create_trajectory(self.T, self.agent, self.env, self.logger)
            final_reward += self.env.curr_value
        final_reward /= test_size
        return final_reward

    def update_final_diff(self):
        for m1, m2 in zip(self.env.synth2copy.modules, self.env.synth.modules):
            for parameter in m1.available_parameters:
                synth2copy_parameter_object = m1.__getattribute__(parameter)
                if not isinstance(synth2copy_parameter_object.value, str):
                    synth_parameter_object = m2.__getattribute__(parameter)
                    diff = synth2copy_parameter_object.value - synth_parameter_object.value
                    log_name = f'{m1.name}_{parameter}' if self.env.hierarchy_type == 'module' else f'{parameter}_{m1.name}'
                    self.logger.update_average(f'{log_name}_diff', abs(diff))
                    self.logger.what2log[f'Final {log_name}_diff'] = diff

    @abstractmethod
    def update_loss(self, episode: UT.Episode = None, batch: list = None):
        pass

    @abstractmethod
    def save_agent_episodes(self, t, episode: UT.Episode):
        pass

    def get_state_representations(self, manager: Env.Manager,  transition: UT.Transition = None):
        transition.to(self.device)
        state1_representation, signal_representation = manager.get_state_representation(transition.state1)
        with torch.no_grad():
            state2_representation, _ = manager.get_state_representation(transition.state2)

        return state1_representation, state2_representation, signal_representation

    def get_critic_error(self, transition: UT.Transition = None, state1_representation=None, state2_representation=None,
                         critic: Env.Critic = None):
        # If is_TD, transition.reward is the immediate reward
        reward = torch.tensor(transition.reward)
        critic_error = reward
        advantage = transition.G
        td_target = transition.G
        if self.is_with_baseline:
            if state1_representation is None or state2_representation is None:
                state1_representation, state2_representation, _ = self.get_state_representations(critic, transition)

            critic_value1 = critic.get_value_of_state(state1_representation)
            self.logger.update_average('critic_value', critic_value1.item())
            critic_error = td_target - critic_value1  # Critic error is always w.r.t G[t]
            # print(transition.G, critic_value1, critic_error, 'G')
            advantage = critic_error.detach()

            # TD estimate instead of Monte Carlo estimate
            if self.is_TD:
                if not transition.terminated:
                    with torch.no_grad():
                        critic_value2 = critic.get_value_of_state(state2_representation)
                else:
                    critic_value2 = 0

                td_target = reward + self.gamma ** transition.n_steps * critic_value2
                advantage = td_target - critic_value1.item()
                critic_error = td_target - critic_value1
                # print(transition.state1[-1], transition.state2[-1])
        return critic_error, advantage, td_target.cpu().numpy()

    def get_advantage(self, transition: UT.Transition = None, state1_representation=None, state2_representation=None):
        if transition.advantage is not None:
            return transition.advantage

        _, advantage, _ = self.get_critic_error(transition, state1_representation, state2_representation,
                                                self.critic.target_critic)
        transition.advantage = advantage.item()
        return transition.advantage

    def update_loss_from_transition(self, transition: UT.Transition = None):
        state1_representation, state2_representation, signal_representation = self.get_state_representations(self.agent,
                                                                                                             transition)

        level = transition.level
        sub_agent = self.agent.get_agent_from_action(transition.action[0], level)
        policy, stop_prob_prev, logits = sub_agent.get_policy_from_state_representation(state1_representation)
        action_log_probability = transition.agent.get_action_log_prob(transition.action, state1_representation,
                                                                      index=level, policy=policy)
        if self.are_shared_backbone:
            advantage = self.get_advantage(transition, state1_representation, state2_representation)
            critic_error, critic_td_error, td_target = self.get_critic_error(transition, state1_representation,
                                                                             state2_representation, self.critic)
        else:
            advantage = self.get_advantage(transition, None, None)
            critic_error, critic_td_error, td_target = self.get_critic_error(transition, None, None, self.critic)

        self.logger.update_average('td_target', td_target)
        transition.td_error = abs(td_target)
        self.calc_loss_components(agent=sub_agent,
                                  action=transition.action,
                                  advantage=advantage,
                                  action_log_prob=action_log_probability,
                                  critic_error=critic_error,
                                  critic_td_error=critic_td_error,
                                  policy=policy,
                                  logits=logits,
                                  transition=transition,
                                  stop_prob=stop_prob_prev,
                                  signal_representation=signal_representation)
        transition.to('cpu')

    def log_data(self, episode: UT.Episode):
        self.update_final_diff()

        self.logger.what2log['Synth Freq'] = self.env.synth2copy.oscillators[0].freq.value
        self.logger.what2log['Synth Cutoff Freq'] = self.env.synth2copy.filters[0].cutoff_freq.value
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                log_name = agent.module_name
                if self.logger.averages[f'{log_name}_dwell'] != 0.0:
                    self.logger.what2log[f'{log_name}_dwell_avg'] = self.logger.averages[f'{log_name}_dwell']

        if self.SRL:
            self.logger.what2log['loss'] = self.logger.averages['param_loss']
            self.logger.what2log['State regression loss'] = self.logger.averages['state_regression_loss']
            self.logger.what2log['Action regression loss'] = self.logger.averages['action_regression_loss']

        # self.logger.what2log[f'{self.agent.module_name} Average advantage'] = self.logger.averages[
        #             f'{self.agent.module_name}_advantage'] * self.env.scale_factor * (-1)

        for t in range(min([len(episode.Rewards), 6])):
            self.logger.what2log[f'Reward time {t}'] = episode.Rewards[t]

        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                self.advantage_logger.what2log[f'{agent.module_name} Average advantage'] = self.advantage_logger.averages[
                    f'{agent.module_name}_advantage']

        self.logger.what2log['Critic error'] = self.logger.averages['critic_error']
        self.logger.what2log['Critic value'] = self.logger.averages['critic_value']

        if self.logger.step > 500 and self.connected2wifi:
            self.logger.what2log['Critic explained variance'] = self.logger.averages['explained_variance']

        if self.logger.step > 100 and self.connected2wifi:
            wandb.log(self.logger.what2log, step=self.logger.step)
            wandb.log(self.advantage_logger.what2log, step=self.logger.step)

        self.logger.what2log['Advantage Cut Off'] = self.logger.averages['advantage_cut_off']
        self.logger.what2log['Advantage'] = self.logger.averages['advantage']
        self.logger.what2log['G0'] = self.logger.averages['G0']

    def print_magnitude(self):
        if self.critic is not None and self.critic.ValueEstimation.value.weight.grad is not None:
            print("Value")
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight, 2)).item()))

        if self.agent.SoundEncoder.features2encoding.output.weight.grad is not None:
            print("Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight, 2)).item()))

        oscillator_agent = self.agent.next_agents[0]
        freq_agent = oscillator_agent.next_agents[0]
        if freq_agent.regression.mu.weight.grad is not None:
            print("Freq mu")
            print(np.sqrt(torch.sum(torch.pow(freq_agent.regression.mu.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(freq_agent.regression.mu.weight, 2)).item()))

        if freq_agent.regression.sigma.weight.grad is not None:
            print("Freq sigma")
            print(np.sqrt(torch.sum(torch.pow(freq_agent.regression.sigma.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(freq_agent.regression.sigma.weight, 2)).item()))

    def zero_agent(self):
        self.agent.episodes = list()

    @abstractmethod
    def after_episode_hook(self, episode: UT.Episode):
        pass


class OnlineAlgorithm(RLOptimizer):
    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: [Env.HierarchicalAgent, Env.SingleActionAgent],
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict,
                 test_set: list,
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):
        super(OnlineAlgorithm, self).__init__(optimizer_agent, optimizer_critic, lr_scheduler_agent,
                                              lr_scheduler_critic, env, agent, critic, are_shared_backbone, is_pretrained,
                                              n_steps, T, n_modules, n_parameters, config, test_set=test_set, run=run,
                                              logger=logger, advantage_logger=advantage_logger)
        self.n_updates = config['n_updates']
        self.trajectory_length = int(self.T / self.n_updates)
        self.k_epochs = 1
        self.n_episodes = int(self.batch_size / T)

    def optimize(self):
        super(OnlineAlgorithm, self).optimize()
        for step in range(self.n_steps):
            print(f"Optimization step {self.logger.opt_step}")

            if self.count_test >= self.test_every or self.count_test == 0:
                test_loss, test_freq_diff, test_cutoff_freq, test_rmse = Env.test_model(self.test_set, self.T, self.env,
                                                                                        self.agent, t=self.run.step)
                self.count_test = 0
                wandb.log({'test_loss': test_loss,
                           'test_freq_diff': test_freq_diff,
                           'test_rmse': test_rmse,
                           'test_cutoff_freq': test_cutoff_freq},
                          step=self.logger.step)

            if isinstance(self.optimizer_agent, UT.AdamTrace):
                self.optimizer_agent.zero_trace()

            if isinstance(self.optimizer_critic, UT.AdamTrace):
                self.optimizer_critic.zero_trace()

            while self.n_steps_for_update < self.batch_size:
            # for n in range(self.n_episodes):
                self.zero_agent()
                self.env.generate_starting_point(self.is_hard_search)
                self.env.update_value()

                episode = UT.Episode()
                state = self.env.get_state()
                episode.States.append(state)

                _, signal_representation = self.critic.get_state_representation(episode.States[-1])
                with torch.no_grad():
                    representation_size = torch.sum(torch.abs(signal_representation))
                self.logger.what2log_t['Critic Representation size vs t'] = representation_size

                Env.create_trajectory(self.trajectory_length, episode, self.agent, self.env, self.logger,
                                      self.connected2wifi)
                self.save_agent_episodes(0, episode)

                episode.FinalReward = self.env.curr_value
                self.update_loss(episode)
                self.update_grads()
                self.agent.episodes = list()

                self.after_episode_hook(episode)
                self.logger.what2log['Final reward'] = episode.FinalReward * self.env.scale_factor
                self.logger.what2log['Episode length'] = len(episode.States)

                self.logger.update_average('final_reward', episode.FinalReward)
                self.logger.update_average('t', len(episode.States))

                self.R.append(episode.FinalReward)

                if step % 100 == 0:
                    self.env.decrease_tolerance()

                if self.logger.step % 250 == 0:
                    self.agent.decrease_temperature()
                torch.cuda.empty_cache()

                self.log_data(episode)
                episode.clear()

            self.count_test += 1
            self.logger.opt_step += 1
            self.update_parameters_wrt_grads()
            self.update_learning_schedulers()
            self.update_batch_advantage_statistics()
            self.n_steps_for_update = 0

        return self.R

    def update_loss(self, episode: UT.Episode = None, batch: list = None):
        Gs, rhos = create_discounted_rewards(episode.Rewards, self.gamma)
        for agent_episode in self.agent.episodes:
            t_start, t_end, _ = agent_episode
            reward = Gs[t_start] - Gs[t_end]
            state1 = copy_nested_list(episode.States[t_start])
            state2 = copy_nested_list(episode.States[t_end])
            action = copy_nested_list(episode.Actions[t_start])
            terminated = True if t_end == len(episode.States) - 1 and episode.terminated else False
            transition = UT.Transition(t_start=t_start,
                                       state1=state1,
                                       state2=state2,
                                       action=action,
                                       reward=reward,
                                       G=Gs[t_start],
                                       agent=self.agent,
                                       level=0,
                                       action_probability=episode.ActionProbabilities[t_start],
                                       time_stamp=self.logger.step,
                                       n_steps=(t_end - t_start),
                                       terminated=terminated)
            self.update_loss_from_transition(transition)
            del transition
            self.batch_count += 1
            self.logger.loss_calc_step += 1
        self.n_steps_for_update += self.batch_count

    def save_agent_episodes(self, t, episode: UT.Episode):
        pass

    def update_parameters_wrt_grads(self, delta=1):
        super(OnlineAlgorithm, self).update_parameters_wrt_grads()
        self.critic.copy_weights_to_target()

    def update_batch_advantage_statistics(self):
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                key = f'{agent.module_name}_advantage'
                if self.advantage_logger.count_for_temporary_average[key] == 0:
                    continue
                batch_mean = (self.advantage_logger.batch_averages[key] /
                                  self.advantage_logger.count_for_temporary_average[key])
                self.advantage_logger.update_average(key, batch_mean)

                batch_mean_squared = (self.advantage_logger.batch_averages_squared[key] /
                                      self.advantage_logger.count_for_temporary_average[key])

                batch_std = (batch_mean_squared - batch_mean ** 2) ** 0.5
                self.advantage_logger.update_average(f'{key}_std', batch_std)

                # Zero the means for the next update
                self.advantage_logger.batch_averages[key] = 0
                self.advantage_logger.batch_averages_squared[key] = 0
                self.advantage_logger.count_for_temporary_average[key] = 0


class OfflineAlgorithm(RLOptimizer):
    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: Env.OptionBasedAgent,
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict,
                 test_set: list,
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):

        super(OfflineAlgorithm, self).__init__(optimizer_agent, optimizer_critic, lr_scheduler_agent,
                                               lr_scheduler_critic, env, agent, critic, are_shared_backbone, is_pretrained,
                                               n_steps, T, n_modules, n_parameters, config, test_set=test_set, run=run,
                                               logger=logger, advantage_logger=advantage_logger)

        self.buffer_size = 2048
        self.ReplayBuffer = Memory(capacity=self.buffer_size, epsilon=0.0001, alpha=1, beta=1)
        # self.dataset = UT.PPOMemory()
        # self.dataloader = DataLoader()
        self.update_target_counter = 0
        self.update_target_every = self.k_epochs * self.batch_size

    def optimize(self):
        super(OfflineAlgorithm, self).optimize()
        decrease_time = self.n_steps * self.T * 10
        for step in range(self.n_steps):
            print(f"Optimization step {self.logger.opt_step}")

            if self.count_test >= self.test_every or self.count_test == 0:
                test_loss, test_freq_diff, test_cutoff_freq, test_rmse = Env.test_model(self.test_set, self.T, self.env,
                                                                                        self.agent, t=self.run.step)
                self.count_test = 0
                wandb.log({'test_loss': test_loss,
                           'test_freq_diff': test_freq_diff,
                           'test_rmse': test_rmse,
                           'test_cutoff_freq': test_cutoff_freq},
                          step=self.logger.step)

            if isinstance(self.optimizer_agent, UT.AdamTrace):
                self.optimizer_agent.zero_trace()

            if isinstance(self.optimizer_critic, UT.AdamTrace):
                self.optimizer_critic.zero_trace()

            self.ReplayBuffer.clear()
            while self.ReplayBuffer.n_entries < self.buffer_size:
                # print(f"Episode {i}")
                self.zero_agent()
                self.env.generate_starting_point(self.is_hard_search)
                self.env.update_value()

                episode = UT.Episode()
                state = self.env.get_state()
                episode.States.append(state)

                _, signal_representation = self.critic.get_state_representation(episode.States[-1])
                with torch.no_grad():
                    representation_size = torch.sum(torch.abs(signal_representation))
                self.logger.what2log_t['Critic Representation size vs t'] = representation_size

                Env.create_trajectory(self.T, episode, self.agent, self.env, self.logger, self.connected2wifi)
                episode.Rewards.append(0)
                self.after_episode_hook(episode)
                self.save_transitions_to_buffer(episode)

                episode.FinalReward = self.env.curr_value
                self.log_data(episode)

                self.logger.what2log['Final reward'] = episode.FinalReward * self.env.scale_factor
                self.logger.what2log['Episode length'] = len(episode.States)

                self.logger.update_average('final_reward', episode.FinalReward)
                self.logger.update_average('t', len(episode.States))

                self.R.append(episode.FinalReward)
                episode.clear()
                self.agent.episodes = list()

            self.count_test += 1
            self.update_weights_in_batches()

            if step % 50 == 0:
                self.env.decrease_tolerance()

            if self.logger.step % int(250 / self.k_epochs) == 0:
                self.agent.decrease_temperature()
            torch.cuda.empty_cache()

            self.logger.opt_step += 1

        return self.R

    def save_transitions_to_buffer(self, episode: UT.Episode):
        episode.to('cpu')
        Gs, rhos = create_discounted_rewards(episode.Rewards, self.gamma)
        for agent_sub_episode in self.agent.episodes:
            t_start, t_end = agent_sub_episode
            reward = Gs[t_start] - Gs[t_end]

            state1 = copy_nested_list(episode.States[t_start])
            state2 = copy_nested_list(episode.States[t_end])
            action = copy_nested_list(episode.Actions[t_start])
            terminated = True if t_end == len(episode.States) - 1 and episode.terminated else False
            transition = UT.Transition(t_start=t_start,
                                       state1=state1,
                                       state2=state2,
                                       action=action,
                                       reward=reward,
                                       G=Gs[t_start],
                                       agent=self.agent,
                                       level=0,
                                       action_probability=episode.ActionProbabilities[t_start],
                                       time_stamp=self.logger.step,
                                       n_steps=(t_end - t_start),
                                       terminated=terminated)
            self.ReplayBuffer.add(transition, transition.td_error)

    @abstractmethod
    def update_weights_in_batches(self, episode: UT.Episode = None):
        pass

    def update_loss(self, episode=None, batch: list = None):
        batch_size_temp = 0
        # transitions, indices, _ = self.ReplayBuffer.sample(batch_size_temp)
        for transition in batch:
            if isinstance(transition, UT.Transition):
                self.update_loss_from_transition(transition)
                batch_size_temp += 1
            # self.ReplayBuffer.update(idx, transition.td_error)
        self.batch_count += batch_size_temp
        self.logger.loss_calc_step += batch_size_temp
        self.update_target_counter += batch_size_temp

    def update_parameters_wrt_grads(self, delta=1):
        super(OfflineAlgorithm, self).update_parameters_wrt_grads()
        if self.update_target_counter >= self.update_target_every:
            self.critic.copy_weights_to_target()
            self.update_target_counter = 0

    def save_agent_episodes(self, t, episode: UT.Episode):
        pass


class Reinforce(OnlineAlgorithm):
    is_single_agent = True

    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: Env.OptionBasedAgent,
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict,
                 test_set: list,
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):
        super(Reinforce, self).__init__(optimizer_agent, optimizer_critic, lr_scheduler_agent, lr_scheduler_critic, env,
                                        agent, critic, are_shared_backbone, is_pretrained, n_steps, T, n_modules, n_parameters,
                                        config, test_set, run=run, logger=logger, advantage_logger=advantage_logger)

    def save_agent_episodes(self, t, episode):
        for t_retro in range(len(episode.States)):
            self.agent.episodes.append([t_retro, len(episode.States) - 1, False])


class ActorCritic(OnlineAlgorithm):
    is_single_agent = True

    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: Env.Manager,
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict = None,
                 test_set: list = [],
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):
        """
            N-step TD Actor Critic algorithm
        """
        super(ActorCritic, self).__init__(optimizer_agent=optimizer_agent,
                                          optimizer_critic=optimizer_critic,
                                          lr_scheduler_agent=lr_scheduler_agent,
                                          lr_scheduler_critic=lr_scheduler_critic,
                                          env=env,
                                          agent=agent,
                                          critic=critic,
                                          are_shared_backbone=are_shared_backbone,
                                          is_pretrained=is_pretrained,
                                          n_steps=n_steps,
                                          T=T,
                                          n_modules=n_modules,
                                          n_parameters=n_parameters,
                                          config=config,
                                          test_set=test_set,
                                          run=run,
                                          logger=logger,
                                          advantage_logger=advantage_logger)

    def save_agent_episodes(self, t, episode):
        for t_retro in range(t, len(episode.Actions)):
            self.agent.episodes.append([t_retro, min([t_retro + self.n_samples, len(episode.States) - 1]), False])

    def after_episode_hook(self, episode: UT.Episode):
        self.update_loss(episode)
        self.update_grads()


class OptionCritic(OnlineAlgorithm):
    is_single_agent = False

    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: [Env.HierarchicalAgent, Env.SingleActionAgent],
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict = None,
                 test_set: list = [],
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):

        super(OptionCritic, self).__init__(optimizer_agent=optimizer_agent,
                                           optimizer_critic=optimizer_critic,
                                           lr_scheduler_agent=lr_scheduler_agent,
                                           lr_scheduler_critic=lr_scheduler_critic,
                                           env=env,
                                           agent=agent,
                                           critic=critic,
                                           are_shared_backbone=are_shared_backbone,
                                           is_pretrained=is_pretrained,
                                           n_steps=n_steps, T=T,
                                           n_modules=n_modules,
                                           n_parameters=n_parameters,
                                           config=config,
                                           test_set=test_set,
                                           run=run,
                                           logger=logger,
                                           advantage_logger=advantage_logger)

    def zero_agent(self):
        self.agent.init_agents()
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                agent.episodes = list()

        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                log_name = agent.module_name
                self.logger.count_for_average[f'{log_name}_dwell'] = 0

    def update_loss(self, episode: UT.Episode = None, batch: list = None):
        Gs, rhos = create_discounted_rewards(episode.Rewards, self.gamma)
        self.logger.update_average('G0', Gs[0])
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                for agent_episode in agent.episodes:
                    t_start, t_end, action_prob, stopped = agent_episode
                    if t_start == t_end or t_start >= self.T:
                        continue
                    reward = Gs[t_start] - Gs[t_end] - self.etta * stopped
                    state1 = copy_nested_list(episode.States[t_start])
                    state2 = copy_nested_list(episode.States[t_end])
                    action = copy_nested_list(episode.Actions[t_start])
                    terminated = True if t_end == len(episode.States) - 1 and episode.terminated else False
                    transition = UT.Transition(t_start=t_start,
                                               state1=state1,
                                               state2=state2,
                                               action=action,
                                               reward=reward,
                                               G=Gs[t_start],
                                               agent=self.agent,
                                               level=level,
                                               action_probability=action_prob,
                                               time_stamp=self.logger.step,
                                               n_steps=(t_end - t_start),
                                               terminated=terminated)
                    self.update_loss_from_transition(transition)
                    del transition
                    self.batch_count += 1
                    self.logger.loss_calc_step += 1
                agent.episodes = list()
        self.n_steps_for_update += self.batch_count

    def print_magnitude(self):
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                print(f"-------- {agent.module_name} --------")
                if agent.regression is not None and agent.regression.mu.weight.grad is not None:
                    print("Regression")
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight, 2)).item()))

                    if agent.regression.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight, 2)).item()))

                if agent.selection is not None and agent.selection.choice.weight.grad is not None:
                    print("Selection")
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight, 2)).item()))

                    if agent.selection.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight, 2)).item()))

        if self.critic is not None and self.critic.ValueEstimation.value.weight.grad is not None:
            print("Value")
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight, 2)).item()))

        if self.critic.SoundEncoder.features2encoding.output.weight.grad is not None:
            print("Critic Representation")
            print(np.sqrt(torch.sum(torch.pow(self.critic.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.SoundEncoder.features2encoding.output.weight, 2)).item()))
            print("Advanced Representation")
            print(np.sqrt(torch.sum(torch.pow(self.critic.advanced_representation.mlp.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.advanced_representation.mlp.output.weight, 2)).item()))

        if self.agent.SoundEncoder.features2encoding.output.weight.grad is not None:
            print("Agent Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight, 2)).item()))
            print("Advanced Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight, 2)).item()))

        if self.agent.advanced_representation.mlp.output.weight.grad is not None:
            print("Adv Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight, 2)).item()))

    def update_parameters_wrt_grads(self, delta=1):
        super(OptionCritic, self).update_parameters_wrt_grads()
        self.critic.copy_weights_to_target()

    def after_episode_hook(self, episode: UT.Episode):
        agent_to_update = self.agent.top_agent
        last_action = episode.Actions[-1][0]
        for idx in last_action:
            # if len(agent_to_update.next_agents) > 1:
            #     agent_to_update.episodes.append([agent_to_update.start_time, len(episode.States) - 1])

            if len(agent_to_update.next_agents) == 0:
                break
            agent_to_update = agent_to_update.next_agents[idx]
        self.update_loss(episode)
        self.update_grads()
        if self.is_hard_search:
            self.check_waveform()
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                dwelling_percent = self.logger.count_for_average[f'{agent.module_name}_dwell'] / len(episode.Actions)
                self.logger.update_average(f'{agent.module_name}_dwell', dwelling_percent)

    def check_waveform(self):
        for oscillator in self.env.synth.oscillators:
            min_value = -100
            arg_min_value = 'sine'
            for waveform in self.env.synth2copy.available_waves:
                oscillator.waveform.value = waveform
                self.env.update_signal(1)
                self.env.update_value()
                if self.env.curr_value >= min_value:
                    min_value = self.env.curr_value
                    arg_min_value = waveform
            oscillator.waveform.value = arg_min_value
            # print(min_value, self.env.synth2copy.oscillators[0].freq.value, self.env.synth.oscillators[0].freq.value, arg_min_value, self.env.synth2copy.oscillators[0].waveform.value)


class PPOC(OfflineAlgorithm):
    is_single_agent = False

    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: [Env.HierarchicalAgent, Env.SingleActionAgent],
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict = None,
                 test_set: list = [],
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):

        super(PPOC, self).__init__(optimizer_agent=optimizer_agent,
                                   optimizer_critic=optimizer_critic,
                                   lr_scheduler_agent=lr_scheduler_agent,
                                   lr_scheduler_critic=lr_scheduler_critic,
                                   env=env,
                                   agent=agent,
                                   critic=critic,
                                   are_shared_backbone=are_shared_backbone,
                                   is_pretrained=is_pretrained,
                                   n_steps=n_steps,
                                   T=T,
                                   n_modules=n_modules,
                                   n_parameters=n_parameters,
                                   config=config,
                                   test_set=test_set,
                                   run=run,
                                   logger=logger,
                                   advantage_logger=advantage_logger)
        self.ReplayBuffer = UT.PPOMemory(capacity=self.buffer_size)  # 4 updates * 4 amount of transitions
        # self.n_episodes = int(self.batch_size / self.T)
        self.advantage_cut_off_count = 0

    def zero_agent(self):
        self.agent.init_agents()
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                agent.episodes = list()
                log_name = agent.module_name
                self.logger.count_for_average[f'{log_name}_dwell'] = 0

    def save_transitions_to_buffer(self, episode: UT.Episode = None):
        episode.to('cpu')
        Gs, rhos = create_discounted_rewards(episode.Rewards, self.gamma)
        self.logger.update_average('G0', Gs[0])
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                for agent_episode in agent.episodes:
                    t_start, t_end, action_prob, stopped = agent_episode
                    if t_start == t_end or t_start >= self.T:
                        continue

                    reward = Gs[t_start] - Gs[t_end] - self.etta * stopped
                    state1 = copy_nested_list(episode.States[t_start])
                    state2 = copy_nested_list(episode.States[t_end])
                    action = copy_nested_list(episode.Actions[t_start])
                    terminated = True if t_end == len(episode.States) - 1 and episode.terminated else False
                    transition = UT.Transition(t_start=t_start,
                                               state1=state1,
                                               state2=state2,
                                               action=action,
                                               reward=reward,
                                               G=Gs[t_start],
                                               agent=self.agent,
                                               level=level,
                                               action_probability=action_prob,
                                               time_stamp=self.logger.step,
                                               n_steps=(t_end - t_start),
                                               terminated=terminated)
                    self.ReplayBuffer.add(transition, transition.td_error)
                agent.episodes = list()

    def update_weights_in_batches(self, episode: UT.Episode = None, update_parameters=False):
        if self.ReplayBuffer.n_entries == 0:
            return
        data_loader = BatchSampler(self.ReplayBuffer, batch_size=self.batch_size, drop_last=False)
        for epoch in range(self.k_epochs):
            for batch in data_loader:
                self.update_loss(batch=batch)
                self.update_grads()
                self.update_parameters_wrt_grads()
        self.update_learning_schedulers()
        self.logger.update_average('advantage_cut_off', self.advantage_cut_off_count / self.ReplayBuffer.n_entries)
        self.advantage_cut_off_count = 0
        self.ReplayBuffer.clear()
        del data_loader

    def after_episode_hook(self, episode: UT.Episode):
        if isinstance(self.agent, Env.CyclicAgent):
            return

        agent_to_update = self.agent.top_agent
        last_action = episode.Actions[-1][0]
        for idx in last_action:
            if len(agent_to_update.next_agents) > 1:
                agent_to_update.episodes.append([agent_to_update.start_time, len(episode.States) - 1,
                                                 torch.exp(agent_to_update.cache_action_log_prob), False])

            if len(agent_to_update.next_agents) == 0:
                break
            agent_to_update = agent_to_update.next_agents[idx]

        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                dwelling_percent = self.logger.count_for_average[f'{agent.module_name}_dwell'] / len(episode.Actions)
                self.logger.update_average(f'{agent.module_name}_dwell', dwelling_percent)

    def print_magnitude(self):
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                print(f"-------- {agent.module_name} --------")
                if agent.regression is not None and agent.regression.mu.weight.grad is not None:
                    print("Regression")
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight, 2)).item()))

                    if agent.regression.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight, 2)).item()))

                if agent.selection is not None and agent.selection.choice.weight.grad is not None:
                    print("Selection")
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight, 2)).item()))

                    if agent.selection.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight, 2)).item()))

        if self.critic is not None and self.critic.ValueEstimation.value.weight.grad is not None:
            print("Value")
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight, 2)).item()))

        if self.agent.SoundEncoder.features2encoding.output.weight.grad is not None:
            print("Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight, 2)).item()))

        if self.agent.advanced_representation.mlp.output.weight.grad is not None:
            print("Adv Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight, 2)).item()))

    def get_actor_loss(self, advantage, action_log_probability, transition):
        rho = torch.exp(action_log_probability) / transition.action_probability
        c = torch.clamp(rho, 1 - self.epsilon, 1 + self.epsilon) * advantage
        if not (1 - self.epsilon <= rho.item(), 1 + self.epsilon):
            self.advantage_cut_off_count += 1
        actor_loss = torch.minimum(rho * advantage, c)
        return -1 * self.RL_weight * actor_loss


class PPO(OfflineAlgorithm):
    is_single_agent = True

    def __init__(self,
                 optimizer_agent: Type[optim.Optimizer],
                 optimizer_critic: Type[optim.Optimizer],
                 lr_scheduler_agent,
                 lr_scheduler_critic,
                 env: Env.Environment,
                 agent: [Env.HierarchicalAgent, Env.SingleActionAgent],
                 critic: Env.Critic,
                 are_shared_backbone: bool,
                 is_pretrained: bool,
                 n_steps: int,
                 T: int,
                 n_modules: int,
                 n_parameters: int,
                 config: Dict = None,
                 test_set: list = [],
                 run: wandb.run = None,
                 logger: UT.Logger = None,
                 advantage_logger: UT.Logger = None):

        super(PPO, self).__init__(optimizer_agent=optimizer_agent,
                                  optimizer_critic=optimizer_critic,
                                  lr_scheduler_agent=lr_scheduler_agent,
                                  lr_scheduler_critic=lr_scheduler_critic,
                                  env=env,
                                  agent=agent,
                                  critic=critic,
                                  are_shared_backbone=are_shared_backbone,
                                  is_pretrained=is_pretrained,
                                  n_steps=n_steps,
                                  T=T,
                                  n_modules=n_modules,
                                  n_parameters=n_parameters,
                                  config=config,
                                  test_set=test_set,
                                  run=run,
                                  logger=logger,
                                  advantage_logger=advantage_logger)
        self.n_episodes = int(self.batch_size / self.T)

    def zero_agent(self):
        self.agent.init_agents()
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                agent.episodes = list()
        self.agent.episodes = list()

    def save_agent_episodes(self, t, episode):
        for t_retro in range(t, len(episode.Actions)):
            self.agent.episodes.append([t_retro, min([t_retro + self.n_samples, len(episode.States) - 1]), False])

    def update_weights_in_batches(self, episode: UT.Episode = None, update_parameters=False):
        data_loader = BatchSampler(self.ReplayBuffer, batch_size=self.batch_size, drop_last=False)
        for epoch in range(self.k_epochs):
            for batch in data_loader:
                self.update_loss(batch=batch)
                self.update_grads()
                self.update_parameters_wrt_grads()
        self.update_learning_schedulers()
        del data_loader

    def after_episode_hook(self, episode: UT.Episode):
        self.save_agent_episodes(0, episode)
        agent_to_update = self.agent.top_agent
        last_action = episode.Actions[-1][0]
        for idx in last_action:
            if len(agent_to_update.next_agents) > 1:
                agent_to_update.episodes.append([agent_to_update.start_time, len(episode.States) - 1, False])

            if len(agent_to_update.next_agents) == 0:
                break
            agent_to_update = agent_to_update.next_agents[idx]

    def print_magnitude(self):
        for level in self.level2agent.keys():
            for agent in self.level2agent[level]:
                print(f"-------- {agent.module_name} --------")
                if agent.regression is not None and agent.regression.mu.weight.grad is not None:
                    print("Regression")
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.regression.mu.weight, 2)).item()))

                    if agent.regression.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.regression.termination.weight, 2)).item()))

                if agent.selection is not None and agent.selection.choice.weight.grad is not None:
                    print("Selection")
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight.grad, 2)).item()))
                    print(np.sqrt(torch.sum(torch.pow(agent.selection.choice.weight, 2)).item()))

                    if agent.selection.termination.weight.grad is not None:
                        print("Stopping")
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight.grad, 2)).item()))
                        print(np.sqrt(torch.sum(torch.pow(agent.selection.termination.weight, 2)).item()))

        if self.critic is not None and self.critic.ValueEstimation.value.weight.grad is not None:
            print("Value")
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.critic.ValueEstimation.value.weight, 2)).item()))

        if self.agent.SoundEncoder.features2encoding.output.weight.grad is not None:
            print("Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.SoundEncoder.features2encoding.output.weight, 2)).item()))

        if self.agent.advanced_representation.mlp.output.weight.grad is not None:
            print("Adv Representation")
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight.grad, 2)).item()))
            print(np.sqrt(torch.sum(torch.pow(self.agent.advanced_representation.mlp.output.weight, 2)).item()))

    def get_actor_loss(self, advantage, action_log_probability, transition):
        rho = torch.exp(action_log_probability) / transition.action_probability
        c = torch.clamp(rho, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = torch.minimum(rho * advantage, c)
        return -1 * self.RL_weight * actor_loss


def create_discounted_rewards(Rewards, gamma_multiplier):
    Discounted_rewards = list()
    rhos = list()
    for t in range(len(Rewards) + 1):
        G = 0
        gamma = 1
        rho_t = 1
        for k, r in enumerate(Rewards[t:]):
            G = G + gamma * r
            gamma = gamma * gamma_multiplier
        Discounted_rewards.append(G)
        rhos.append(rho_t)
    return Discounted_rewards, rhos


def copy_nested_list(original_list):
    if isinstance(original_list, list):
        # If the input is a list, recursively copy each item
        return [copy_nested_list(item) for item in original_list]
    else:
        # If the input is not a list, return the item as it is (base case)
        return original_list


algorithm2class = {"ActorCritic": ActorCritic,
                   "OptionCritic": OptionCritic,
                   "Reinforce": Reinforce,
                   "PPOC": PPOC,
                   "PPO": PPO}
