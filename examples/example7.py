import cvxpy as cp
import mocvxpy as mocp
import numpy as np
from matplotlib import pyplot as plt

# Taken from
#
# Ehrgott, M., Shao, L., & Schöbel, A. (2011).
# An approximation algorithm for convex multi-objective programming problems.
# Journal of Global Optimization, 50(3), p. 397-416.
# https://doi.org/10.1007/s10898-010-9588-7
#
# Solve:
# min f(x) = [50 x1^4 + 10 x2^4,
#             30 (x1 - 5)^4 + 100 (x2 - 3)^4,
#             70 (x1 - 2)^4 + 20 (x2 - 4)^4]
# s.t. (x1 - 2)^2 + (x2 - 2)^2 <= 1
#      0 <= x1 <= 3
#      0 <= x2 <= 3
x = mocp.Variable(2)

objectives = [
    cp.Minimize(50 * x[0] ** 4 + 10 * x[1] ** 4),
    cp.Minimize(30 * (x[0] - 5) ** 4 + 100 * (x[1] - 3) ** 4),
    cp.Minimize(70 * (x[0] - 2) ** 4 + 20 * (x[1] - 4) ** 4),
]
constraints = [x >= 0, x <= 3, cp.sum_squares(x - 2 * np.ones(2)) <= 1]

pb = mocp.Problem(objectives, constraints)

objective_values = pb.solve(
    solver="MONMO",
)
print("status: ", pb.status)

# TODO: using MOSEK solver for solving pascoletti-serafini scalarization
# and gurobi as qp solver, results on a segmentation fault on this
# example. The problem comes from pycddlib, when calling matrix_canonicalize.
#
# Using OSQP results on a earlier termination.
objective_values = pb.solve(
    solver="MOVS",
    vertex_selection_solver_options={"solver": cp.CLARABEL},
)
print("status: ", pb.status)

objective_values = pb.solve(
    solver="ADENA",
)
print("status: ", pb.status)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
    [vertex[2] for vertex in objective_values],
)
plt.show()
