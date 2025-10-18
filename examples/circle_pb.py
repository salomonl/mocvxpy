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

# Solve:
# min f(x) = [x1, x2]
# s.t. (x1 - 1)**2 + (x2 - 1)**2 <= 1
#      x1, x2 >= 0
n = 2

x = mocp.Variable(n)

objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]

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

ax = plt.figure().add_subplot()
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
)
plt.show()
