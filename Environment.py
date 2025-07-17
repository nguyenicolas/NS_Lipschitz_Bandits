import numpy as np

class Environment:
    def __init__(self, T, nb_shifts, centers=None, shift_noise_std=0.01):
        self.T = T
        self.nb_shifts = nb_shifts
        self.phase_length = T // nb_shifts
        self.dyna_regret = []

        if centers is None:
            self.centers = np.linspace(0.1, 0.9, nb_shifts)
        else:
            assert len(centers) == nb_shifts, "Length of centers must match nb_shifts"
            self.centers = centers

        # Optional small perturbation (adversarial or stochastic shift noise)
        self.shifts = np.random.normal(0, shift_noise_std, size=T)

    def mean_reward(self, t, x):
        phase = min(t // self.phase_length, self.nb_shifts - 1)
        center = self.centers[phase] #+ self.shifts[t]

        width = 0.2
        height = 0.3

        distance = abs(x - center)
        if distance < width:
            return 0.5 + height * (1 - distance / width)
        else:
            return 0.5

    def get_reward(self, t, x):
        mu_t = self.mean_reward(t, x)
        return int(np.random.rand() < mu_t)

    def cumulative_dyna_regret(self, t, x_t, best_value_t=0.8):
        regret_t = best_value_t - self.mean_reward(t, x_t)
        self.dyna_regret.append(regret_t)


