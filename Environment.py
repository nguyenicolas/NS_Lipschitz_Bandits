class Environment :
    def __init__(self, T, nb_shifts) -> None:
        self.T = T
        self.nb_shifts = nb_shifts
        self.dyna_regret = []

    def mean_reward(self, t):
        # Nave test, just for sanity check for now
        def naive_function(x):
            #width = 0.3
            #center = 0.5
            #distance = abs(x - center)
            #if distance < width:
            #    return 1 - distance / width
            #else:
            #    return 0.0
            if t <= 1600 :
                return x
            return 1 - x
        return naive_function
    
    def get_reward(self, t, x):
        return self.mean_reward(t)(x)
    
    def cumulative_dyna_regret(self, t, x_t):
        best_value_t = 1 # assumuming this pour l'instant, sinon trop galère
        regret_t = best_value_t - self.mean_reward(t).naive_function(x_t)
        self.dyna_regret.append(regret_t)
