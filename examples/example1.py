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
constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]

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
