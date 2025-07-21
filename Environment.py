import numpy as np

class Environment:
    def __init__(self, T, nb_shifts, centers=None, shift_noise_std=0.01):
        self.T = T
        self.nb_shifts = nb_shifts
        self.phase_length = T // nb_shifts
        self.dyna_regret = []

        if centers is None:
            self.centers = [0.2, 0.8, 0.2, 0.8, 0.2]
            #self.centers = self._generate_alternating_centers(nb_shifts)
            print( "CENTERSSSS = ", self.centers)
            
        else:
            assert len(centers) == nb_shifts, "Length of centers must match nb_shifts"
            
            

        # Optional small perturbation (adversarial or stochastic shift noise)
        self.shifts = np.random.normal(0, shift_noise_std, size=T)

    def _generate_alternating_centers(self, nb_shifts):
        """Generate well-separated centers in an alternating pattern."""
        base_centers = [0.2, 0.2, 0.8, 0.8] #np.linspace(0.2, 0.8, nb_shifts)
        reordered = []
        left = 0
        right = nb_shifts - 1
        while left <= right:
            reordered.append(base_centers[left])
            if left != right:
                reordered.append(base_centers[right])
            left += 1
            right -= 1
        return reordered[:nb_shifts]

    def mean_reward(self, t, x):
        phase = min(t // self.phase_length, self.nb_shifts - 1)
        center = self.centers[phase]  # You can add + self.shifts[t] if needed

        width = 0.2
        height = 0.5

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
