# Code for Non-Stationary Lipschitz Bandits

Anonymous repo for the Neurips submission #1681 : Non-stationary Lipschitz Bandits

## Repository Structure

``` shell
├── Dyadic.py        # Code for binary tree structure
├── Policy.py        # Code for bandit policis
├── Environment.py   # Code for bandit environment
├── Results/         # Numerical results
```

## Synthetic Experiments

### Bandit Environment
We simulate a non-stationary bandit environment over a time horizon of `T = 100,000`, where the mean reward function evolves over time. In particular, we have $L_T=\mathcal{O}(T)$ and $V_T = \mathcal{O}(T)$. The environment uses a mean reward that cyclically shift between two distinct optima:
- **Peak at `x₁ = 0.3`**
- **Peak at `x₂ = 0.7`**
  
This transition occurs **10 times** throughout the time horizon. The mean reward function at each time step is **1-Lipschitz** with respect to the action space `x ∈ [0, 1]`

Below are animations illustrating the evolution of the mean reward function over time. Each frame shows the reward as a function of the action space `x` at a particular time step.

![Mean reward](mean_reward_evolution.gif)


### Numerical results
We plot the dynamic regret of our algorithm 'MBDE', with the standard confidence intervals averaged over 'NB_ITER=100' iterations.
We consider 2 benchmarks:
- 'Binning+UCB (Naive)': discretizes the space into $K(T)=\mathcal{O}(T^{1/3})$ actions, then run UCB naively over these $K(T)$ arms.
- 'Binning+UCB (Oracle)': knows when the shift occurs. For each significant pahse $[\tau_i, \tau_{i+1}[$ (see definition 2 of our submission pdf), it discretizes the space into $K_i=\mathcal{O}((\tau_{i+1}-\tau_i)^{1/3})$ arms and run UCB with these $K_i$ arms on each phase. At the end of each phase, it discards all estimates.
