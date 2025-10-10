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
x = mocp.Variable(2)

# Parameters
a = np.array([[1, 1], [2, 3], [4, 2]])

objectives = [
    cp.Minimize(cp.sum_squares(x - a[0])),
    cp.Minimize(cp.sum_squares(x - a[1])),
    cp.Minimize(cp.sum_squares(x - a[2])),
]
constraints = [x >= 0, x <= [10, 4], x[0] + 2 * x[1] <= 10]

pb = mocp.Problem(objectives, constraints)

objective_values = pb.solve(
    solver="MONMO", scalarization_solver_options={"solver": cp.MOSEK}
)
print("status: ", pb.status)

objective_values = pb.solve(
    solver="MOVS",
    scalarization_solver_options={"solver": cp.MOSEK},
    vertex_selection_solver_options={"solver": cp.GUROBI},
)
print("status: ", pb.status)

objective_values = pb.solve(
    solver="ADENA", scalarization_solver_options={"solver": cp.MOSEK}
)
print("status: ", pb.status)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
    [vertex[2] for vertex in objective_values],
)
plt.show()
