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
