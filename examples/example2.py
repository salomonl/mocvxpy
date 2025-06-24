import cvxpy as cp
import numpy as np
import mocvxpy as mocp

from matplotlib import pyplot as plt

# Solve:
# min f(x) = [(x1 - 3)^2 + (x2 - 3)^2
#             (x1 - 1)^2 + (x2 - 1)^2]
# s.t. |x1| + 2 |x2| <= 2
x = mocp.Variable(2)

objectives = [cp.Minimize(cp.sum_squares(x - np.array([3., 3.]))),
              cp.Minimize(cp.sum_squares(x - np.array([1., 1.])))]
constraints = [cp.abs(x[0]) + 2 * cp.abs(x[1]) <= 2]

solver = mocp.MONMOSolver(objectives, constraints)
status, solution = solver.solve()

solver = mocp.MOSVSolver(objectives, constraints)
status, solution = solver.solve()

ax = plt.figure().add_subplot()
ax.scatter([vertex[0] for vertex in solution.objective_values],
           [vertex[1] for vertex in solution.objective_values])
plt.show()
