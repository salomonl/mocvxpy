import cvxpy as cp
import numpy as np
import mocvxpy as mocp

from dask.distributed import Client
from matplotlib import pyplot as plt

# Solve:
# min f(x) = [(x1 - 3)^2 + (x2 - 3)^2
#             (x1 - 1)^2 + (x2 - 1)^2]
# s.t. |x1| + 2 |x2| <= 2
if __name__ == "__main__":
    x = mocp.Variable(2)

    objectives = [
        cp.Minimize(cp.sum_squares(x - np.array([3.0, 3.0]))),
        cp.Minimize(cp.sum_squares(x - np.array([1.0, 1.0]))),
    ]
    constraints = [cp.abs(x[0]) + 2 * cp.abs(x[1]) <= 2]

    pb = mocp.Problem(objectives, constraints)

    client = Client()
    objective_values = pb.solve(
        client=client, solver="MONMO", scalarization_solver_options={"solver": cp.MOSEK}
    )
    print("status: ", pb.status)

    ax = plt.figure().add_subplot()
    ax.scatter(
        [vertex[0] for vertex in objective_values],
        [vertex[1] for vertex in objective_values],
    )
    plt.show()
