import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import torch.optim as optim
from typing import List, Optional, Callable
from torch import Tensor
import csv
import matplotlib.pyplot as plt
import math


mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,
    n_fft=16384,
    hop_length=512,
    n_mels=128,
    power=1.0,  # Magnitude
    normalized=True,
    f_min=20
)

class Episode(object):
    def __init__(self,
                 States=list(),
                 Actions=list(),
                 Rewards=list(),
                 ActionProbabilities=list(),
                 FinalReward=0,
                 terminated=False):

        self.States = States
        self.Actions = Actions
        self.Rewards = Rewards
        self.ActionProbabilities = ActionProbabilities
        self.FinalReward = -FinalReward  # For minimum heap
        self.terminated = terminated

    def clear(self):
        delete_iterable_items(self.States)
        delete_iterable_items(self.Actions)
        delete_iterable_items(self.Rewards)
        delete_iterable_items(self.ActionProbabilities)
        self.FinalReward = 0

    def __lt__(self, other):
        return self.FinalReward < other.FinalReward

    def to(self, device):
        temp_states = list()
        temp_actions = list()
        temp_actions_probabilities = list()
        for state in self.States:
            temp_states.append([state_element.to(device) for state_element in state])

        if isinstance(self.Actions[0][0], list):
            for action in self.Actions:
                action_temp = list()
                for sub_action in action:
                    action_temp.append([action_element.to(device) for action_element in sub_action])
                temp_actions.append(action_temp)
        else:
            for action in self.Actions:
                temp_actions.append([action_element.to(device) for action_element in action])

        for action_probability in self.ActionProbabilities:
            temp_actions_probabilities.append(action_probability.to(device))

        temp_rewards = self.Rewards.copy()
        self.clear()
        self.States = temp_states
        self.Actions = temp_actions
        self.Rewards = temp_rewards
        self.ActionProbabilities = temp_actions_probabilities


class Transition(object):
    def __init__(self,
                 t_start,
                 state1,
                 state2,
                 action,
                 reward,
                 G,
                 agent,
                 td_error=100,
                 action_probability=torch.tensor(1),
                 level=2,
                 time_stamp=0,
                 n_steps=1,
                 terminated=False):

        self.t_start = t_start
        self.state1 = state1
        self.state2 = state2
        self.action = action
        self.reward = reward
        self.G = G
        self._td_error = -td_error  # For minimum heap
        self.action_probability = action_probability
        self.agent = agent
        self.level = level
        self.time_stamp = time_stamp
        self.terminated = terminated
        self.n_steps = n_steps
        self.advantage = None

    @property
    def td_error(self):
        return self._td_error

    @td_error.setter
    def td_error(self, new_value):
        self._td_error = -1 * new_value

    def __lt__(self, other):
        return self.td_error < other.td_error

    def __del__(self):
        delete_iterable_items(self.state1)
        delete_iterable_items(self.state2)
        delete_iterable_items(self.action)
        delete_iterable_items(self.action_probability)
        del self._td_error
        del self

    def to(self, device):
        self.state1 = [state_element.to(device) for state_element in self.state1]
        self.state2 = [state_element.to(device) for state_element in self.state2]

        if isinstance(self.action[0], list):
            action_temp = list()
            for sub_action in self.action:
                action_temp.append([action_element.to(device) for action_element in sub_action])
            self.action = action_temp
        else:
            self.action = [action_element.to(device) for action_element in self.action]
        self.action_probability = self.action_probability.to(device)


class Logger(object):
    def __init__(self,
                 average_type,
                 moving_avg_alpha):

        self.average_type = average_type
        self.moving_avg_alpha = moving_avg_alpha
        self.averages = dict()
        self.averages_squared = dict()
        self.std = dict()
        self.count_for_average = dict()

        self.what2log_t = dict()
        self.what2log = dict()
        self.step = 0
        self.loss_calc_step = 0
        self.opt_step = 0

    # Update average of a specific logged variable
    def update_average(self, average, new_value, add_to_count=True):
        if add_to_count:
            self.count_for_average[average] += 1
        if self.average_type == 'regular':
            self.update_regular_average(average, new_value)
        elif self.average_type == 'ewma':
            self.update_ewma(average, new_value)

    def update_regular_average(self, average, new_value):
        # Update running mean and variance based on Welford's algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        diff_from_mean1 = new_value - self.averages[average]
        self.averages[average] = self.averages[average] + diff_from_mean1 / self.count_for_average[average]
        diff_from_mean2 = new_value - self.averages[average]
        self.averages_squared[average] = self.averages_squared[average] + diff_from_mean1 * diff_from_mean2
        self.std[average] = np.sqrt(self.averages_squared[average]) / self.count_for_average[average]

    def update_ewma(self, average, new_value):
        # Update running mean and variance based on EWMA
        # https://en.wikipedia.org/wiki/Moving_average
        # https://stats.stackexchange.com/questions/111851/standard-deviation-of-an-exponentially-weighted-mean
        diff_from_mean1 = new_value - self.averages[average]
        self.averages[average] = self.moving_avg_alpha * self.averages[average] +\
                                 (1 - self.moving_avg_alpha) * new_value
        self.averages_squared[average] = (1 - self.moving_avg_alpha) * (self.averages_squared[average] +
                                                                        self.moving_avg_alpha * np.power(diff_from_mean1, 2))
        # self.averages_squared[average] = self.moving_avg_alpha * self.averages_squared[average] +\
        #                                  (1 - self.moving_avg_alpha) * np.power(diff_from_mean1, 2)
        # self.std[average] = np.sqrt(self.averages_squared[average]) / self.count_for_average[average]
        self.std[average] = np.sqrt(self.averages_squared[average])

    def set_up_log(self, what2log):
        self.averages[what2log] = 0
        self.averages_squared[what2log] = 0
        self.std[what2log] = 1
        self.count_for_average[what2log] = 0


class PPOMemory(Dataset):
    def __init__(self, capacity):
        self.transitions = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.capacity = capacity

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        x = self.transitions[index]
        return x

    def add(self, x, priority=0):
        self.transitions[self.n_entries % self.capacity] = x
        self.n_entries += 1

    def clear(self):
        self.transitions = np.zeros(self.capacity, dtype=object)
        self.n_entries = 0


class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, total_warmup_steps, total_zero_steps=0, max_steps=10000, with_decay=False):
        self.total_steps_zero = total_zero_steps
        self.total_warmup_steps = total_warmup_steps
        self.max_steps = max_steps
        self.with_decay = with_decay
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_steps_zero:
            if self.last_epoch > self.total_warmup_steps:
                if self.with_decay:
                    return [base_lr * (self.max_steps - self.last_epoch) / self.max_steps for base_lr in self.base_lrs]
                return self.base_lrs
            return [base_lr * (self.last_epoch - self.total_steps_zero) / self.total_warmup_steps for base_lr in self.base_lrs]
        else:
            return [0 for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        return super(GradualWarmupScheduler, self).step(epoch)


class AdamTrace(optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.99, gamma=1, eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)
        super(AdamTrace, self).__init__(params=params, defaults=defaults)

        self.init_state()

        self.gamma = gamma

    def init_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['trace'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['trace_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['max_trace_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def zero_trace(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, delta=1, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

                Args:
                    delta: delta for multiplying the gradient
                    closure (callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            traces = []
            trace_avg_sqs = []
            max_trace_avg_sqs = []
            state_steps = []
            beta1 = group['beta1']
            beta2 = group['beta2']
            gamma = self.gamma

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    traces.append(state['trace'])
                    trace_avg_sqs.append(state['trace_avg_sq'])

                    if group['amsgrad']:
                        max_trace_avg_sqs.append(state['max_trace_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            rmstrace(params_with_grad,
                     grads,
                     traces,
                     trace_avg_sqs,
                     max_trace_avg_sqs,
                     state_steps,
                     delta=delta,
                     amsgrad=group['amsgrad'],
                     beta1=beta1,
                     beta2=beta2,
                     gamma=gamma,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'],
                     maximize=group['maximize'])
        return loss


def rmstrace(params: List[Tensor],
             grads: List[Tensor],
             traces: List[Tensor],
             trace_avg_sqs: List[Tensor],
             max_trace_avg_sqs: List[Tensor],
             state_steps: List[int],
             *,
             delta: float,
             amsgrad: bool,
             beta1: float,
             beta2: float,
             gamma: float,
             lr: float,
             weight_decay: float,
             eps: float,
             maximize: bool):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        trace = traces[i]
        trace_avg_sq = trace_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        trace_avg_sq.mul_(beta2).addcmul_(grad, grad.conj())
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_trace_avg_sqs[i], trace_avg_sq, out=max_trace_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_trace_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (trace_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = (lr * delta) / bias_correction1
        trace.mul_(beta1).addcdiv_(grad, denom, value=gamma)
        param.add_(trace, alpha=-step_size)


def delete_iterable_items(iterable):
    if isinstance(iterable, list):
        while len(iterable) > 0:
            if isinstance(iterable[-1], list):
                delete_iterable_items(iterable.pop())
            else:
                iterable.pop()

    elif isinstance(iterable, dict):
        for k in list(iterable.keys()):
            if isinstance(iterable[k], list):
                delete_iterable_items(iterable[k])
            else:
                del iterable[k]

    else:
        del iterable


def get_random_agents(top_agent):
    current_agent = top_agent
    agents_indices = list()
    while len(current_agent.next_agents) > 0:
        next_agent_index = np.random.choice(range(len(current_agent.next_agents)))
        agents_indices.append(torch.tensor(next_agent_index))
        current_agent = current_agent.next_agents[next_agent_index]
    return current_agent, agents_indices


def create_test_set_list(test_set_dir):
    with open(test_set_dir) as tf:
        reader = csv.reader(tf)
        entire_list = list()
        for row in reader:
            row_list = list()
            for value in row:
                if value.find('.') == -1:
                    row_list.append(int(value))
                else:
                    row_list.append(float(value))
            entire_list.append(row_list)
        test_set = [(entire_list[i], entire_list[i + 1]) for i in range(0, len(entire_list), 2)]
    return test_set


def collate_fn(batch):
    print(batch)


def earth_mover_distance(y_true, y_pred):
    y_pred_cumsum0 = torch.cumsum(y_pred, dim=0)
    y_true_cumsum0 = torch.cumsum(y_true, dim=0)
    square = torch.square(y_true_cumsum0 - y_pred_cumsum0)
    final = torch.mean(square)
    return final


def magnitude_penalized_emd(y_true, y_pred):
    sum1 = torch.sum(y_pred, dim=0)
    sum2 = torch.sum(y_true, dim=0)
    #
    # max_magnitude1 = torch.max(y_pred, dim=0)
    # max_magnitude2 = torch.max(y_true, dim=0)
    # magnitude_diff = torch.abs(max_magnitude1[0] - max_magnitude2[0])
    # # magnitude_penalty = torch.pow(magnitude_diff + 1, 2) - 1

    y_pred = y_pred / sum1
    y_true = y_true / sum2

    y_pred_cumsum0 = torch.cumsum(y_pred, dim=0)
    y_true_cumsum0 = torch.cumsum(y_true, dim=0)

    absolute = torch.abs(y_true_cumsum0 - y_pred_cumsum0)
    absolute = torch.sum(absolute, dim=0)
    # final = torch.mean(absolute) + 10 * torch.sum(magnitude_diff)
    final = torch.mean(absolute)
    return final


def earth_mover_distance_averaged(y_true, y_pred, kernel_size, is_plot=False):
    n_pools = int((kernel_size + 1) / 2)
    avg_pools = list()
    for i in range(n_pools):
        avg_pool = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=i)
        avg_pools.append(avg_pool)

    if is_plot:
        fig, axes = plt.subplots(2)
        axes[0].imshow(y_true.cpu().numpy())
        axes[1].imshow(y_pred.cpu().numpy())
        plt.show()

    y_pred_avgs = list()
    y_true_avgs = list()
    for i in range(n_pools):
        y_pred_avg = avg_pools[i](y_pred.T).T
        y_true_avg = avg_pools[i](y_true.T).T

        sum1 = torch.sum(y_pred_avg, dim=0)
        sum2 = torch.sum(y_true_avg, dim=0)
        y_pred_avgs.append(y_pred_avg / sum1)
        y_true_avgs.append(y_true_avg / sum2)

        # y_pred_avgs.append(y_pred_avg)
        # y_true_avgs.append(y_true_avg)

    min_val = torch.tensor([128], device='cuda')
    for i in range(n_pools):
        for j in range(n_pools):
            # Slicing the results of the pools for dimension equality
            if y_pred_avgs[i].shape[0] > y_true_avgs[j].shape[0]:
                y_pred_sliced = y_pred_avgs[i][:y_true_avgs[j].shape[0], :]
                y_true_sliced = y_true_avgs[j]

            elif y_pred_avgs[i].shape[0] < y_true_avgs[j].shape[0]:
                y_pred_sliced = y_pred_avgs[i]
                y_true_sliced = y_true_avgs[j][:y_pred_avgs[i].shape[0], :]

            else:
                y_pred_sliced = y_pred_avgs[i]
                y_true_sliced = y_true_avgs[j]

            y_pred_sliced = torch.cumsum(y_pred_sliced, dim=0)
            y_true_sliced = torch.cumsum(y_true_sliced, dim=0)

            if is_plot:
                fig, ax = plt.subplots()
                ax.plot(range(y_pred.shape[0]), y_pred_sliced[:, 0].cpu().numpy())
                ax.plot(range(y_pred.shape[0]), y_true_sliced[:, 0].cpu().numpy())
                for s in range(1, 345):
                    ax.plot(range(y_pred.shape[0]), y_pred_sliced[:, s: s + 1].cpu().numpy())
                    ax.plot(range(y_pred.shape[0]), y_true_sliced[:, s: s + 1].cpu().numpy())
                    fig.canvas.draw()
                    plt.pause(0.1)
                    plt.cla()
                plt.show()

            curr_emd = torch.abs(y_pred_sliced - y_true_sliced)
            curr_emd = torch.sum(curr_emd, dim=0)
            curr_emd = torch.mean(curr_emd)
            if curr_emd.item() <= min_val.item():
                min_val = curr_emd
    return min_val
