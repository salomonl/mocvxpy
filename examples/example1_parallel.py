import cvxpy as cp
import mocvxpy as mocp

from dask.distributed import Client
from matplotlib import pyplot as plt

# Solve:
# min f(x) = [x1, x2]
# s.t. (x1 - 1)**2 + (x2 - 1)**2 <= 1
#      x1, x2 >= 0
if __name__ == "__main__":
    n = 2

    x = mocp.Variable(n)

    objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
    constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]

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
