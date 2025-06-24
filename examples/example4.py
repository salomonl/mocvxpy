import cvxpy as cp
import mocvxpy as mocp
import numpy as np

from matplotlib import pyplot as plt

# Solve:
# min f(x) = [exp(x1) + exp(x4),
#             exp(x2) + exp(x5),
#             exp(x3) + exp(x6)]
# s.t. x1 + x2 + x3 >= 0
#      3 x1 + 6 x2 + 3 x3 + 4 x4 + x5 + x6 >= 0
#      3 x2 + x2 + x3 + 2 x4 + 4 x5 + 4 x6 >= 0
n = 6
x = mocp.Variable(n)
objectives = [cp.Minimize(cp.exp(x[0]) + cp.exp(x[3])),
              cp.Minimize(cp.exp(x[1]) + cp.exp(x[4])),
              cp.Minimize(cp.exp(x[2]) + cp.exp(x[5]))]
constraints = [np.array([[1, 1, 1, 0, 0, 0],
                         [3, 6, 3, 4, 1, 1],
                         [3, 1, 1, 2, 4, 4]]) @ x >= 0]

solver = mocp.MOSVSolver(objectives, constraints)
status, solution = solver.solve()

ax = plt.figure().add_subplot(projection="3d")
ax.scatter([vertex[0] for vertex in solution.objective_values],
           [vertex[1] for vertex in solution.objective_values],
           [vertex[2] for vertex in solution.objective_values])
plt.show()
