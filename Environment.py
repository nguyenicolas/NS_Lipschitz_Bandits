import numpy as np

class Environment :
    def __init__(self, T, nb_shifts) -> None:
        self.T = T
        self.nb_shifts = nb_shifts
        self.dyna_regret = []
        self.shifts = np.random.normal(0.02, 0.01, size=(self.T, ))

    def mean_reward(self, t, x):
        # Alternate every 2000 rounds between 0.3 and 0.7
        if (t // 3000) % 2 == 0:
            center = 0.2
        else:
            center = 0.8

        center += self.shifts[t]  # Optional perturbation

        width = 0.1    # Width of the bump
        height = 0.3   # Max bump height above baseline 0.5

        distance = abs(x - center)
        if distance < width:
            return 0.5 + height * (1 - distance / width)
        else:
            return 0.5
        
    def get_reward(self, t, x):
        mu_t = self.mean_reward(t, x)
        return int(np.random.rand() < mu_t)
    
    def cumulative_dyna_regret(self, t, x_t):
        best_value_t = 0.8 # assumuming this pour l'instant, sinon trop galère
        regret_t = best_value_t - self.mean_reward(t, x_t)
        self.dyna_regret.append(regret_t)



