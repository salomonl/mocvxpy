import cvxpy as cp
import mocvxpy as mocp
import numpy as np

from dask.distributed import Client
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
if __name__ == "__main__":
    x = mocp.Variable(2)

    objectives = [
        cp.Minimize(50 * x[0] ** 4 + 10 * x[1] ** 4),
        cp.Minimize(30 * (x[0] - 5) ** 4 + 100 * (x[1] - 3) ** 4),
        cp.Minimize(70 * (x[0] - 2) ** 4 + 20 * (x[1] - 4) ** 4),
    ]
    constraints = [x >= 0, x <= 3, cp.sum_squares(x - 2 * np.ones(2)) <= 1]

    pb = mocp.Problem(objectives, constraints)

    client = Client()
    objective_values = pb.solve(
        client=client,
        solver="MONMO",
    )
    print("status: ", pb.status)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(
        [vertex[0] for vertex in objective_values],
        [vertex[1] for vertex in objective_values],
        [vertex[2] for vertex in objective_values],
    )
    plt.show()
