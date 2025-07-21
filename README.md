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

### Non-stationary Bandit Environment

Below are animations of the mean rewards at different time steps.

![WTA Training](images/sgd_wta.gif)
*Winner-takes-all training dynamics with stochastic gradient descent (see Fig.1)*

![aMCL Training](images/sgd_amcl.gif)
*Annealed Multiple Choice Learning training dynamics with stochastic gradient descent (see Fig.1)*

### Numerical results
