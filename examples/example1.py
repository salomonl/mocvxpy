import cvxpy as cp
import mocvxpy as mocp

from matplotlib import pyplot as plt

# Solve:
# min f(x) = [x1, x2]
# s.t. (x1 - 1)**2 + (x2 - 1)**2 <= 1
#      x1, x2 >= 0
n = 2

x = mocp.Variable(n)

objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
constraints = [x >= 0,
               cp.sum_squares(x - 1) <= 1]

solver = mocp.MONMOSolver(objectives, constraints)
status, solution = solver.solve()

solver = mocp.MOSVSolver(objectives, constraints)
status, solution = solver.solve()

# pb = mocp.Problem(objectives, constraints)

ax = plt.figure().add_subplot()
ax.scatter([vertex[0] for vertex in solution.objective_values],
           [vertex[1] for vertex in solution.objective_values])
plt.show()
