class Environment :
    def __init__(self, T, nb_shifts) -> None:
        self.T = T
        self.nb_shifts = nb_shifts

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
            return x
        return naive_function
    
    def get_reward(self, t, x):
        return self.mean_reward(t)(x)