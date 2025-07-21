import cvxpy as cp
import mocvxpy as mocp
import numpy as np

import matplotlib as mpl

mpl.use("macosx")
from matplotlib import pyplot as plt

# Solve:
# min f(x) = x with respect to the order cone C
# s.t. || x - 1 ||_2 <= 1
# x in R^2
n = 3
x = mocp.Variable(n)

objectives = [cp.Minimize(x[i]) for i in range(n)]
constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
# Compute the cone defined by:
# C = conv cone {(1, 2)^T, (2, 1)^T}
# C = mocp.compute_order_cone_from_its_rays(np.array([[1, 2], [2, 1]]))
# C = mocp.compute_order_cone_from_its_rays(np.array([[2, -1], [-1, 2]]))

# C = mocp.compute_order_cone_from_its_rays(np.array([[4, 2, 2],
#                                                     [2, 4, 2],
#                                                     [4, 0, 2],
#                                                     [1, 0, 2],
#                                                     [0, 1, 2],
#                                                     [0, 4, 2]]))

C = mocp.compute_order_cone_from_its_rays(
    np.array([[-1, -1, 3], [2, 2, -1], [1, 0, 0], [0, -1, 2], [-1, 0, 2], [0, 1, 0]])
)

pb = mocp.Problem(objectives, constraints, C)

objective_values = pb.solve(
    solver="MOVS",
    scalarization_solver_options={"solver": cp.MOSEK},
    vertex_selection_solver_options={"solver": cp.GUROBI},
)
print("status: ", pb.status)

objective_values = pb.solve(
    solver="MONMO", scalarization_solver_options={"solver": cp.MOSEK}
)
print("status: ", pb.status)

# ax = plt.figure().add_subplot()
# ax.scatter([vertex[0] for vertex in solution.objective_values],
#            [vertex[1] for vertex in solution.objective_values])
# plt.show()

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
    [vertex[2] for vertex in objective_values],
)
plt.show()
