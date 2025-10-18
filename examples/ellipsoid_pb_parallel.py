"""
Copyright 2025 Ludovic Salomon, Daniel Dörfler and Andreas Löhne.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cp
import matplotlib as mpl

mpl.use("macosx")
import mocvxpy as mocp

from dask.distributed import Client
from matplotlib import pyplot as plt

# Solve:
# min f(x) = [x1, x2, x3]
# s.t. (x1 - 1)**2 + ((x2 - 1) / a)**2 + ((x3 - 1)) / 5)**2 <= 1
if __name__ == "__main__":
    a = 7.0
    n = 3
    x = mocp.Variable(n)

    objectives = [cp.Minimize(x[0]), cp.Minimize(x[1]), cp.Minimize(x[2])]
    constraints = [(x[0] - 1) ** 2 + ((x[1] - 1) / a) ** 2 + ((x[2] - 1) / 5) ** 2 <= 1]

    pb = mocp.Problem(objectives, constraints)

    client = Client()
    objective_values = pb.solve(
        client=client, solver="MONMO", scalarization_solver_options={"solver": cp.MOSEK}
    )
    print("status: ", pb.status)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(
        [vertex[0] for vertex in objective_values],
        [vertex[1] for vertex in objective_values],
        [vertex[2] for vertex in objective_values],
    )
    plt.show()
