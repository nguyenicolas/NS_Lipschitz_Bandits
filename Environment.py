import numpy as np

class Environment :
    def __init__(self, T, nb_shifts) -> None:
        self.T = T
        self.nb_shifts = nb_shifts
        self.dyna_regret = []

    def mean_reward(self, t):
        # Nave test, just for sanity check for now
        def bump(x):
            width = 0.3       # width of the bump
            center = 0.5      # center of the bump
            height = 0.3      # maximum height above 0.5

            distance = abs(x - center)
            if distance < width:
                return 0.5 + height * (1 - distance / width)
            else:
                return 0.5

        return bump

    
    def get_reward(self, t, x):
        #return self.mean_reward(t)(x)
        mu_t = self.mean_reward(t)(x)
        return int(np.random.rand() < mu_t)
    
    def cumulative_dyna_regret(self, t, x_t):
        best_value_t = 0.8 # assumuming this pour l'instant, sinon trop galère
        regret_t = best_value_t - self.mean_reward(t)(x_t)
        self.dyna_regret.append(regret_t)
