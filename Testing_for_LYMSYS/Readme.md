# Testing for LYMSYS Dataset

This experiment applies our test to the LYMSYS chatbot leaderboard dataset, a widely used benchmark for evaluating the performance of Large Language Models (LLMs).

## Code Organization

* `Testing_for_Lymsys.ipynb`: Main code file that generates the output files and includes a plotting section.
* `resultsconcat.csv`: Output file containing test statistics for the Bradley-Terry-Luce (BTL) model for various choices of parameters n and k.
* `resultsTconcat.csv`: Output file containing test statistics for the Thurstone (Case V) model for various choices of parameters n and k.

## Running the Experiment

1. Recommended to run on Google Colab (free tier works).
2. Open `Testing_for_Lymsys.ipynb` in Google Colab.
3. Run all cells to generate the output files and plots.

## Output Files

* `resultsconcat.csv` and `resultsTconcat.csv` contain the test statistics for the BTL and Thurstone models, respectively, for various choices of parameters n and k.
* The plotting section in `Testing_for_Lymsys.ipynb` generates visualizations of the results.
