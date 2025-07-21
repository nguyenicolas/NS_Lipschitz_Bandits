import numpy as np

class Environment:
    def __init__(self, T, num_cycles):
        self.T = T
        self.num_cycles = num_cycles
        self.dyna_regret = []
        self.max_reward_over_time()

    def mean_reward(self):
        def tent_peak(x, center, width=2.0):
            dist = abs(x - center)
            if dist >= width / 2:
                return 0.0
            else:
                return 1.0 - (2.0 * dist / width)  # slope ≤ 1

        def reward(t, x):
            x = min(max(x, 0.0), 1.0)

            # Normalize t to the number of cycles
            # e.g., if num_cycles = 10 and T = 100, this is 10 full sine periods over T
            cycle_pos = (t / self.T) * self.num_cycles  # e.g., 0 to 10
            alpha = 0.5 * (1 + np.sin(2 * np.pi * cycle_pos - np.pi / 2))  # in [0,1]

            # f0: peak at 0.3; f1: peak at 0.7
            f0 = tent_peak(x, center=0.3, width=2.0)
            f1 = tent_peak(x, center=0.7, width=2.0)

            return (1 - alpha) * f0 + alpha * f1

        return reward

    def max_reward_over_time(self, resolution=100):
        """
        Returns the maximum value of the reward function at each time t in [0, T-1].
        """
        reward_fn = self.mean_reward()
        x_vals = np.linspace(0, 1, resolution)
        max_values = np.zeros(self.T)

        for t in range(self.T):
            rewards = [reward_fn(t, x) for x in x_vals]
            max_values[t] = max(rewards)

        self.max_values = max_values

    def get_reward(self, t, x):
        mu_t = self.mean_reward()(t, x)
        return int(np.random.rand() < mu_t)
