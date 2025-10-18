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
import numpy as np
import mocvxpy as mocp

from dask.distributed import Client
from matplotlib import pyplot as plt

# Example 5.2 taken from
#
# Ehrgott, M., Shao, L., & Schöbel, A. (2011).
# An approximation algorithm for convex multi-objective programming problems.
# Journal of Global Optimization, 50(3), p. 397-416.
# https://doi.org/10.1007/s10898-010-9588-7
#
# Solve:
# min f(x) = [(x1 - 3)^2 + (x2 - 3)^2
#             (x1 - 1)^2 + (x2 - 1)^2]
# s.t. |x1| + 2 |x2| <= 2
if __name__ == "__main__":
    # Create problem
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.sum_squares(x - np.array([3.0, 3.0]))),
        cp.Minimize(cp.sum_squares(x - np.array([1.0, 1.0]))),
    ]
    constraints = [cp.abs(x[0]) + 2 * cp.abs(x[1]) <= 2]
    pb = mocp.Problem(objectives, constraints)

    # Create a Client instance: must be initialized into a function
    # Solve problem with MONMO algorithm
    client = Client()
    objective_values = pb.solve(client=client, solver="MONMO")
    print("status: ", pb.status)

    # Plot solutions in the objective space
    ax = plt.figure().add_subplot()
    ax.scatter(
        [vertex[0] for vertex in objective_values],
        [vertex[1] for vertex in objective_values],
    )
    plt.xlabel("$f_1$")
    plt.ylabel("$f_2$")
    plt.show()
