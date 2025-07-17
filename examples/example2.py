import cvxpy as cp
import numpy as np
import mocvxpy as mocp

from matplotlib import pyplot as plt

# Solve:
# min f(x) = [(x1 - 3)^2 + (x2 - 3)^2
#             (x1 - 1)^2 + (x2 - 1)^2]
# s.t. |x1| + 2 |x2| <= 2
x = mocp.Variable(2)

objectives = [
    cp.Minimize(cp.sum_squares(x - np.array([3.0, 3.0]))),
    cp.Minimize(cp.sum_squares(x - np.array([1.0, 1.0]))),
]
constraints = [cp.abs(x[0]) + 2 * cp.abs(x[1]) <= 2]

pb = mocp.Problem(objectives, constraints)

objective_values = pb.solve(solver="MONMO")
print("status: ", pb.status)

objective_values = pb.solve(solver="MOVS")
print("status: ", pb.status)

objective_values = pb.solve(solver="ADENA")
print("status: ", pb.status)

ax = plt.figure().add_subplot()
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
)
plt.show()
