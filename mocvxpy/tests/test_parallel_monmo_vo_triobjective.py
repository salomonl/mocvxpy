import cvxpy as cp
import mocvxpy as mocp
import numpy as np

from .config_dask_client import CLIENT


def test_solve_sphere_pb_respect_to_C1_with_MONMO():
    C1 = mocp.compute_order_cone_from_its_rays(
        np.array([[4, 2, 2], [2, 4, 2], [4, 0, 2], [1, 0, 2], [0, 1, 2], [0, 4, 2]])
    )
    n = 3
    x = mocp.Variable(n)
    objectives = [cp.Minimize(x[i]) for i in range(n)]
    constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
    pb = mocp.Problem(objectives, constraints, C1)

    objective_values = pb.solve(client=CLIENT, solver="MONMO")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (356, 3)
    assert x.values.shape == (356, 3)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    assert np.all(x.values == objective_values)


def test_solve_sphere_pb_respect_to_C2_with_MONMO():
    C2 = mocp.compute_order_cone_from_its_rays(
        np.array(
            [[-1, -1, 3], [2, 2, -1], [1, 0, 0], [0, -1, 2], [-1, 0, 2], [0, 1, 0]]
        )
    )
    n = 3
    x = mocp.Variable(n)
    objectives = [cp.Minimize(x[i]) for i in range(n)]
    constraints = [cp.sum_squares(x - np.ones(n)) <= 1]
    pb = mocp.Problem(objectives, constraints, C2)

    objective_values = pb.solve(client=CLIENT, solver="MONMO")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (356, 3)
    assert x.values.shape == (356, 3)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    assert np.all(x.values == objective_values)
