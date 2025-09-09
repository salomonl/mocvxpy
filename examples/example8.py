import cvxpy as cp
import mocvxpy as mocp

from matplotlib import pyplot as plt

# Taken from
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
x = mocp.Variable(3)

objectives = [
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 10 * x[1] - 120 * x[2]),
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 80 * x[0] - 448 * x[1] + 80 * x[2]),
    cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 448 * x[0] + 80 * x[1] + 80 * x[2]),
]
constraints = [x >= 0, x <= 10, cp.sum_squares(x) <= 1]

pb = mocp.Problem(objectives, constraints)

objective_values = pb.solve(
    solver="MONMO",
)
print("status: ", pb.status)

objective_values = pb.solve(
    solver="MOVS",
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
