# Non-Stationary Lipschitz Bandits
Code and numerical experiments for the paper [Non-Stationary Lipschitz Bandits](https://arxiv.org/abs/2505.18871).

Nicolas Nguyen, Solenne Gacuher, Claire Vernade.


## Abstract
We study the problem of non-stationary Lipschitz bandits, where the number of actions is infinite and the reward function, satisfying a Lipschitz assumption, can change arbitrarily over time. We design an algorithm that adaptively tracks the recently introduced notion of significant shifts, defined by large deviations of the cumulative reward function. To detect such reward changes, our algorithm leverages a hierarchical discretization of the action space. Without requiring any prior knowledge of the non-stationarity, our algorithm
achieves a minimax-optimal dynamic regret bound of  $\mathcal{\widetilde{O}}(\tilde{L}^{1/3}T^{2/3})$, where $\tilde{L}$ is the number of significant shifts and $T$ the horizon. This result provides the first optimal guarantee in this setting.

## Repository Structure
todo

## Contact
If any question contact nguyennicolasviet@gmail.com.
