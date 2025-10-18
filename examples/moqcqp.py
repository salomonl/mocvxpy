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

from matplotlib import pyplot as plt

# Example 5.10 taken from
#
# Ehrgott, M., Shao, L., & Schöbel, A. (2011).
# An approximation algorithm for convex multi-objective programming problems.
# Journal of Global Optimization, 50(3), p. 397-416.
# https://doi.org/10.1007/s10898-010-9588-7
#
# Solve:
# min f(x) = [x1^2 + x2^2 + x3^2 + 10 x2 - 120 x3,
#             x1^2 + x2^2 + x3^2 + 80 x1 - 448 x2 + 80 x3,
#             x1^2 + x2^2 + x3^2 - 448 x1 + 80 x2 + 80 x3]
# s.t. x1^2 + x2^2 + x3^2 <= 100
#      0 <= x <= 10

# Create problem
x = mocp.Variable(3)
objectives = [
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 10 * x[1] - 120 * x[2]),
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 80 * x[0] - 448 * x[1] + 80 * x[2]),
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 448 * x[0] + 80 * x[1] + 80 * x[2]),
]
constraints = [x >= 0, x <= 10, cp.sum_squares(x) <= 1]
pb = mocp.Problem(objectives, constraints)

# Solve problem with MONMO solver
objective_values = pb.solve(
    solver="MONMO",
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
