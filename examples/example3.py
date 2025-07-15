import cvxpy as cp
import mocvxpy as mocp
import numpy as np
from matplotlib import pyplot as plt

# Solve:
# min f(x) = [(x - a[1])^T (x - a[1])
#             (x - a[2])^T (x - a[2])
#             (x - a[3])^T (x - a[3])]
# s.t. x1 + 2 x2 <= 10
#      0 <= x1 <= 10
#      0 <= x2 <= 4
# with a[1] = (1, 1)^T
#      a[2] = (2, 3)^T
#      a[3] = (4, 2)^T
x = mocp.Variable(2)

# Parameters
a = np.array([[1, 1], [2, 3], [4, 2]])

objectives = [cp.Minimize(cp.sum_squares(x - a[0])),
              cp.Minimize(cp.sum_squares(x - a[1])),
              cp.Minimize(cp.sum_squares(x - a[2]))]
constraints = [x >= 0, x <= [10, 4], x[0] + 2 * x[1] <= 10]

# solver = mocp.MONMOSolver(objectives, constraints)
# status, solution = solver.solve()

# solver = mocp.MOVSSolver(objectives, constraints)
# status, solution = solver.solve()

solver = mocp.ADENASolver(objectives, constraints)
status, solution = solver.solve()

ax = plt.figure().add_subplot(projection='3d')
ax.scatter([vertex[0] for vertex in solution.objective_values],
            [vertex[1] for vertex in solution.objective_values],
            [vertex[2] for vertex in solution.objective_values])
plt.show()
