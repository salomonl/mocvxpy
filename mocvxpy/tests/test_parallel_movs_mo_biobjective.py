import cvxpy as cp
import mocvxpy as mocp
import numpy as np

from .config_dask_client import CLIENT


def test_solve_circle_pb_with_MOVS():
    x = mocp.Variable(2)
    objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
    constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(client=CLIENT, max_iter=100)
    assert pb.status == "scalarization_pb_numeric"
    assert objective_values.shape == (62, 2)
    assert x.values.shape == (62, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(x.values == objective_values)


def test_solve_norm1_min_st_qp_constraints():
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.norm(x, 1)),
        cp.Minimize(cp.norm(x - np.array([2, 0]), 1)),
    ]
    constraints = [cp.sum_squares(x) <= 100]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(client=CLIENT, max_iter=100)
    assert pb.status == "optimal"
    assert objective_values.shape == (3, 2)
    assert x.values.shape == (3, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    x_values = x.values
    assert np.all(objective_values[:, 0] == np.sum(np.abs(x_values), axis=1))
    assert np.all(
        objective_values[:, 1] == np.sum(np.abs(x_values - [2.0, 0.0]), axis=1)
    )


def test_solve_qp_with_linear_constraints_with_MOVS():
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.sum_squares(x - np.array([3.0, 3.0]))),
        cp.Minimize(cp.sum_squares(x - np.array([1.0, 1.0]))),
    ]
    constraints = [cp.abs(x[0]) + 2 * cp.abs(x[1]) <= 2]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(client=CLIENT, max_iter=10)
    assert pb.status == "iteration_limit"
    assert objective_values.shape == (118, 2)
    assert x.values.shape == (118, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    x_values = x.values
    assert np.all(
        objective_values[:, 0] == np.sum((x_values - np.array([3.0, 3.0])) ** 2, axis=1)
    )
    assert np.all(
        objective_values[:, 1] == np.sum((x_values - np.array([1.0, 1.0])) ** 2, axis=1)
    )
