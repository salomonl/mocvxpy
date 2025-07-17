import cvxpy as cp
import mocvxpy as mocp

import matplotlib as mpl

mpl.use("macosx")
from matplotlib import pyplot as plt

# Solve:
# min f(x) = [x1, x2, x3]
# s.t. (x1 - 1)**2 + ((x2 - 1) / a)**2 + ((x3 - 1)) / 5)**2 <= 1
a = 7.0
n = 3
x = mocp.Variable(n)

objectives = [cp.Minimize(x[0]), cp.Minimize(x[1]), cp.Minimize(x[2])]
constraints = [(x[0] - 1) ** 2 + ((x[1] - 1) / a) ** 2 + ((x[2] - 1) / 5) ** 2 <= 1]

pb = mocp.Problem(objectives, constraints)

objective_values = pb.solve(solver="MOVS")
print("status: ", pb.status)

objective_values = pb.solve(solver="MONMO")
print("status: ", pb.status)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(
    [vertex[0] for vertex in objective_values],
    [vertex[1] for vertex in objective_values],
    [vertex[2] for vertex in objective_values],
)
plt.show()
