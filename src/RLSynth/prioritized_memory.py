import random
import numpy as np
from src.RLSynth.SumTree import SumTree


class Memory:
    def __init__(self, capacity, epsilon, alpha, beta):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.start_beta = beta
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, sample, error):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def clear(self):
        self.tree.clear()
        self.beta = self.start_beta
