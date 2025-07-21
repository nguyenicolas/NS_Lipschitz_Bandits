- faudrait regler le pb de float division by 0
- faudrait peut etre proceder comme solenne a dit, cad intercaler les replays: peut etre que ca mène à - d'exploration

- WARNING : j'ai rajouté le log(T) dans le treshold

## SETTING 1

nb_shifts = 3
phase_length = 4000
T = nb_shifts*phase_length
env = Environment.Environment(T, nb_shifts)

algo_MBDE = Policy.MBDE(T, c0=1.5)
algo_BinningUCB = Policy.BinningUCB(T)
algo_BinningUCB_Oracle = Policy.BinningUCB_Oracle(T, nb_shifts)



def _generate_alternating_centers(self, nb_shifts):
    """Generate well-separated centers in an alternating pattern."""
    base_centers = np.linspace(0.2, 0.8, nb_shifts)
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



## SETTING 2

nb_shifts = 2
phase_length = 4000
T = nb_shifts*phase_length
env = Environment.Environment(T, nb_shifts)

algo_MBDE = Policy.MBDE(T, c0=1.)
algo_BinningUCB = Policy.BinningUCB(T)
algo_BinningUCB_Oracle = Policy.BinningUCB_Oracle(T, nb_shifts)

sim = Simulator(
    algos={
        "BinningUCB": algo_BinningUCB,
        "BinningUCB (Oracle)": algo_BinningUCB_Oracle,
        "MBDE": algo_MBDE,
    },
    env=env,
    T=T
)


## SETTING 3
alterner 0.2, 0.8 sur des intervalles de 4000. je pense que 4000 est le bon regime d'oublie

ca c'est un serting qui rend UCB binning nul : 
nb_shifts = 6
phase_length = 5000