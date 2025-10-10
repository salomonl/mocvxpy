"""
Copyright 2025 Ludovic Salomon, Daniel Dörfler and Andreas Löhne.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cp
import mocvxpy as mocp
import numpy as np
import operator
import pandas as pd

from matplotlib import pyplot as plt

# Problem adapted from the pymoo test cases
# See:
# https://www.pymoo.org/case_studies/portfolio_allocation.html

if __name__ == "__main__":
    # Load portfolio data
    df = pd.read_csv(
        "./data/portfolio_allocation.csv", parse_dates=True, index_col="date"
    )

    # Compute the portfolio returns, the means and covariances
    returns = df.pct_change().dropna(how="all")
    mu = (1 + returns).prod() ** (252 / returns.count()) - 1
    cov = returns.cov() * 252

    mu, cov = mu.to_numpy(), cov.to_numpy()

    # Define the problem
    nvars = len(df.columns)
    x = mocp.Variable(nvars)  # The percentage to invest on each portfolio

    exp_return = x @ mu
    # We do not consider sqrt(x^T cov x)
    exp_risk = cp.quad_form(x, cov)

    constraints = [
        0 <= x,
        x <= 1,  # They are percentages
        cp.sum(x) == 1,  # The invested sum cannot be bigger than 100%
    ]
    objectives = [cp.Minimize(exp_risk), cp.Minimize(-exp_return)]
    pb = mocp.Problem(objectives, constraints)

    # Solve problem
    objective_values = pb.solve()

    # Convert to standard deviations
    opt_exp_risks = np.sqrt(objective_values[:, 0])

    # We want the maximum expected returns
    opt_exp_returns = -objective_values[:, 1]

    # Find the portfolio with the maximum sharpe
    risk_free_rate = 0.02
    sharpe = (opt_exp_returns - risk_free_rate) / opt_exp_risks
    sharpe_max_ind = sharpe.argmax()
    sharpe_max_weights = np.copy(x.values[sharpe_max_ind])

    # Some investments are too small to make any sense (for example, when
    # inferior to 1e-3). Remove them. Keep sum wi = 0.
    sharpe_max_weights[sharpe_max_weights < 1e-3] = 0
    sharpe_max_weights = sharpe_max_weights / np.sum(sharpe_max_weights)

    allocation = {name: w for name, w in zip(df.columns, sharpe_max_weights)}
    allocation = sorted(allocation.items(), key=operator.itemgetter(1), reverse=True)
    print("Allocation strategy with best sharpe:")
    for name, w in allocation:
        print(f"{name:<5} {w}")

    ax = plt.figure().add_subplot()
    ax.scatter(
        opt_exp_risks,
        opt_exp_returns,
        facecolor="none",
        edgecolors="blue",
        alpha=0.7,
        label="Pareto-Optimal portfolio",
    )
    ax.scatter(
        opt_exp_risks[sharpe_max_ind],
        opt_exp_returns[sharpe_max_ind],
        marker="x",
        s=80,
        color="red",
        label="Max Sharpe Portfolio",
    )
    ax.scatter(
        cov.diagonal() ** 0.5,
        mu,
        facecolor="none",
        edgecolors="black",
        s=30,
        label="Asset",
    )
    plt.legend()
    plt.xlabel("expected volatility")
    plt.ylabel("expected return")
    plt.show()
