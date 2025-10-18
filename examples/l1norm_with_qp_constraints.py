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

from matplotlib import pyplot as plt

# Solve:
# min f(x) = [|x1| + |x2|,
#             |x1 - 2| + |x2|]
# s.t. x1^2 + x2^2 <= 100
x = mocp.Variable(2)

objectives = [cp.Minimize(cp.norm(x, 1)), cp.Minimize(cp.norm(x - np.array([2, 0]), 1))]
constraints = [cp.sum_squares(x) <= 100]

pb = mocp.Problem(objectives, constraints)

# NB: The Pareto front has a "flat" shape, algorithms
# like MOVS or MONMO find its anchor points,
# which can be confusing for practitioners.
# ADENA computes more points, but it helps
# the user to visualize the set of solutions
# without requiring the use of a polyhedron library
objective_values = pb.solve(
    solver="ADENA",
)
print("status: ", pb.status)

ax = plt.figure().add_subplot()
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
)
plt.show()
