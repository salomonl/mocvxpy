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

from matplotlib import pyplot as plt

# Example 5.3 taken from
#
# Löhne, A., Rudloff, B., & Ulus, F. (2014).
# Primal and dual approximation algorithms for convex vector optimization problems.
# Journal of Global Optimization, 60(4), 713-736.
# https://doi.org/10.1007/s10898-013-0136-0
#
# Solve:
# min f(x) = x with respect to the order cone C
# s.t. || x - 1 ||_2 <= 1
# x in R^3

# Create problem
n = 3
x = mocp.Variable(n)
objectives = [cp.Minimize(x[i]) for i in range(n)]
constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
C = mocp.compute_order_cone_from_its_rays(
    np.array([[-1, -1, 3], [2, 2, -1], [1, 0, 0], [0, -1, 2], [-1, 0, 2], [0, 1, 0]])
)
pb = mocp.Problem(objectives, constraints, C)

# Solve problem with MOVS algorithm
objective_values = pb.solve(
    solver="MOVS",
    scalarization_solver_options={"solver": cp.CLARABEL},
    vertex_selection_solver_options={"solver": cp.CLARABEL},
)
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
