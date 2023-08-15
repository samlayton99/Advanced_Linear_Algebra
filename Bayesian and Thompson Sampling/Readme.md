# Bayesian Statistics and Thompson Sampling

In this project, I primarily explore statistical applications of the following two concepts. 

**'Bayesian Sampling.ipynb'** uses Bayes' theorem to update probability estimates for a hypothesis as new data is obtained. It combines prior beliefs with observed data to produce a revised probability. Unlike frequentist statistics, which gives point estimates, Bayesian approaches offer probability distributions over outcomes. This method is valuable in situations with scarce or uncertain data, allowing for adaptable predictions and decision-making.

**Thompson Sampling.py** probabilistically selects actions based on their posterior distributions, balancing exploration and exploitation. It's favored for its adaptability in online learning scenarios.


## Function overview of Thompson Sampling

1. **Thompson Sampling**
    - `thompson(theta, N)`: Utilizes the Thompson Sampling algorithm to simulate arm pulls based on success probabilities (`theta`). Provides an evolving belief about each arm's success through simulation and returns the cumulative success percentages.

2. **Traditional A/B Testing**
    - `abtest(thetas, N, m)`: Executes traditional A/B testing, initially testing each arm `m` times. Post-testing, the arm with the highest success rate gets selected for subsequent pulls. Outputs the aggregated success rate.

3. **Alternative A/B Testing**
    - `abtest2(thetas, N, m)`: An altered A/B testing approach where arms are picked randomly during the test phase. Once tested, the best-performing arm is picked for the remaining rounds. Offers a cumulative success rate.

4. **Comparative Assessment of Rewards**
    - `newassement(thetas, N, m, beta)`: Pits the Thompson Sampling and alternative A/B test rewards against each other. Incorporates a discount factor (`beta`) to weigh earlier successes more heavily.

5. **Holistic Reward Evaluation**
    - `experiment(thetas, N, m, beta, trials)`: A comprehensive assessment that calls `newassement()` multiple times. Returns average rewards across all iterations to provide a balanced understanding of performance.

6. **Variable Rewards with Thompson Sampling**
    - `altthompson(J, theta, N)`: An iteration of Thompson Sampling that integrates varying rewards. Arms grant different rewards from the `J` list based on success. Outputs the average expected return for each pull.

## Functions and Mathematical Background of Bayesian Sampling

The Bayesian model leverages the Normal and Gamma probability distributions. The Normal distribution is characterized by its mean (mu) and variance (sigma^2), while the Gamma distribution is defined by its shape (alpha) and rate (beta). 

1. **Height Estimation Function**
    - `height_model(observed_heights, m, s, alpha, beta, n)`: This function leverages a PyMC3 model to estimate the mean height from a given dataset. The observed heights are assumed to follow a Normal distribution centered around the mean height of the population.
    
2. **The Bayesian Modeling Framework**
    - Our underlying model makes certain assumptions about the distribution of observed data. This section delves deep into the statistical foundations of the `height_model` function, elucidating how the priors for `mu` and `tau` are formulated.

3. **Sampling Techniques using MCMC**
    - Bayesian inference typically involves deriving the posterior distribution. To do so, Markov Chain Monte Carlo (MCMC) methods are employed. These methods are intricate but pivotal for extracting meaningful insights from the data.

4. **Visual Interpretations using Trace Plots**
    - After the MCMC sampling, it's crucial to visualize the drawn samples. Trace plots provide a snapshot of these values over iterations, acting as a diagnostic tool for the convergence of the algorithm.

5. **Expected Value Computation**
    - An essential step in Bayesian modeling, this phase involves computing the expected value of the posterior distribution to obtain point estimates.

## How to Use

1. Load your dataset containing observed heights.
2. Choose appropriate prior parameters for the model.
3. Implement the `height_model` function to extract statistical insights.
4. Visualize results using trace plots.
5. Extract the expected value of the posterior to understand the estimated mean height.

## Dependencies

- pymc3
- numpy
- arviz

## Sample Implementation

To demonstrate how to employ the `height_model` function using PyMC3:

```python
import pymc3 as pm
import numpy as np
import arviz as az
