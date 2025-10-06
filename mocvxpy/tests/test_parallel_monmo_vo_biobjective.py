import cvxpy as cp
import mocvxpy as mocp
import numpy as np

from .config_dask_client import CLIENT


def test_solve_disc_pb_respect_to_C1_with_MONMO():
    # C1 = conv cone {(1, 2)^T, (2, 1)^T}
    C1 = mocp.compute_order_cone_from_its_rays(np.array([[1, 2], [2, 1]]))

    n = 2
    x = mocp.Variable(n)
    objectives = [cp.Minimize(x[i]) for i in range(n)]
    constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
    pb = mocp.Problem(objectives, constraints, C1)

    objective_values = pb.solve(client=CLIENT, solver="MONMO")
    assert pb.status == "optimal"
    assert objective_values.shape == (129, 2)
    assert x.values.shape == (129, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(x.values == objective_values)


def test_solve_disc_pb_respect_to_C2_with_MONMO():
    # C2 = conv cone {(2, -1)^T, (-1, 2)^T}
    C2 = mocp.compute_order_cone_from_its_rays(np.array([[2, -1], [-1, 2]]))

    n = 2
    x = mocp.Variable(n)
    objectives = [cp.Minimize(x[i]) for i in range(n)]
    constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
    pb = mocp.Problem(objectives, constraints, C2)

    objective_values = pb.solve(client=CLIENT, solver="MONMO")
    assert pb.status == "optimal"
    assert objective_values.shape == (33, 2)
    assert x.values.shape == (33, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(x.values == objective_values)
