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

from dask.distributed import Client
from matplotlib import pyplot as plt

# Example 5.8 taken from
#
# Ehrgott, M., Shao, L., & Schöbel, A. (2011).
# An approximation algorithm for convex multi-objective programming problems.
# Journal of Global Optimization, 50(3), p. 397-416.
# https://doi.org/10.1007/s10898-010-9588-7
#
# Solve:
# min f(x) = [(x - a[1])^T (x - a[1])
#             (x - a[2])^T (x - a[2])
#             (x - a[3])^T (x - a[3])]
# s.t. x1 + 2 x2 <= 10
#      0 <= x1 <= 10
#      0 <= x2 <= 4
# with a[1] = (1, 1)^T
#      a[2] = (2, 3)^T
#      a[3] = (4, 2)^T
if __name__ == "__main__":
    # Parameters
    a = np.array([[1, 1], [2, 3], [4, 2]])

    # Create problem
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.sum_squares(x - a[0])),
        cp.Minimize(cp.sum_squares(x - a[1])),
        cp.Minimize(cp.sum_squares(x - a[2])),
    ]
    constraints = [x[0] >= 1, x[1] >= 0, x <= [10, 4], x[0] + 2 * x[1] <= 10]
    pb = mocp.Problem(objectives, constraints)

    # Create a Client instance: must be initialized into a function
    # Solve problem with MONMO algorithm
    client = Client()
    objective_values = pb.solve(client=client, solver="MONMO")
    print("status: ", pb.status)

    # Plot solutions in the objective space
    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(
        [vertex[0] for vertex in objective_values],
        [vertex[1] for vertex in objective_values],
        [vertex[2] for vertex in objective_values],
    )
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_zlabel("$f_3$")
    plt.show()
