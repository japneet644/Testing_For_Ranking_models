# Synthetic Experiment 2: Confidence Intervals under Null Hypothesis

This experiment focuses on approximating confidence intervals under the null hypothesis for the Bradley-Terry-Luce (BTL) model.

## Objective

* Estimate the constants in Proposition 4.
* Approximate the distribution of ∥Fˆ − F∥F.

## Methodology

* Estimating constant c7 in Proposition 4:
	+ Plot several trajectories of the normalized stochastic process.
	+ Identify the 95th quantile for the stochastic process.
* Estimating the quantile of $∥F − Fˆ∥_F$:
	+ Utilize the asymptotic normality of vector $\Delta = \hat{w} − w^∗$.
	+ Compute wˆ as in (7) and estimate $∥F − Fˆ∥_F$.

## Code Organization

* `CombinedCode.ipynb`: Main file for running the experiment.

## Running the Experiment

1. Install required dependencies: `numpy`, `matplotlib`, `scipy`.
2. Run `CombinedCode.ipynb` to generate plots and estimates.


