import cvxpy as cp
import mocvxpy as mocp
import numpy as np


def test_solve_qp_st_linear_constraints_with_ADENA():
    # Parameters
    a = np.array([[1, 1], [2, 3], [4, 2]])
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.sum_squares(x - a[0])),
        cp.Minimize(cp.sum_squares(x - a[1])),
        cp.Minimize(cp.sum_squares(x - a[2])),
    ]
    constraints = [x >= 0, x <= [10, 4], x[0] + 2 * x[1] <= 10]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(solver="ADENA")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (353, 3)
    assert x.values.shape == (353, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    x_values = x.values
    assert np.all(objective_values[:, 0] == np.sum((x_values - a[0]) ** 2, axis=1))
    assert np.all(objective_values[:, 1] == np.sum((x_values - a[1]) ** 2, axis=1))
    assert np.all(objective_values[:, 2] == np.sum((x_values - a[2]) ** 2, axis=1))


def test_solve_ellipsoidal_pb_with_ADENA():
    # Parameters
    a = 7.0
    x = mocp.Variable(3)
    objectives = [cp.Minimize(x[0]), cp.Minimize(x[1]), cp.Minimize(x[2])]
    constraints = [(x[0] - 1) ** 2 + ((x[1] - 1) / a) ** 2 + ((x[2] - 1) / 5) ** 2 <= 1]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(solver="ADENA")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (353, 3)
    assert x.values.shape == (353, 3)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    assert np.all(objective_values == x.values)


def test_solve_power_of_4_pb_st_qp_constraints_with_ADENA():
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(50 * x[0] ** 4 + 10 * x[1] ** 4),
        cp.Minimize(30 * (x[0] - 5) ** 4 + 100 * (x[1] - 3) ** 4),
        cp.Minimize(70 * (x[0] - 2) ** 4 + 20 * (x[1] - 4) ** 4),
    ]
    constraints = [x >= 0, x <= 3, cp.sum_squares(x - 2 * np.ones(2)) <= 1]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(solver="ADENA")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (353, 3)
    assert x.values.shape == (353, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    x_values = x.values
    assert np.all(
        objective_values[:, 0] == 50 * x_values[:, 0] ** 4 + 10 * x_values[:, 1] ** 4
    )
    assert np.all(
        objective_values[:, 1]
        == 30 * (x_values[:, 0] - 5) ** 4 + 100 * (x_values[:, 1] - 3) ** 4
    )
    assert np.all(
        objective_values[:, 2]
        == 70 * (x_values[:, 0] - 2) ** 4 + 20 * (x_values[:, 1] - 4) ** 4
    )


def test_solve_qcqp_with_ADENA():
    x = mocp.Variable(3)

    objectives = [
        cp.Minimize(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 10 * x[1] - 120 * x[2]),
        cp.Minimize(
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 80 * x[0] - 448 * x[1] + 80 * x[2]
        ),
        cp.Minimize(
            x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 448 * x[0] + 80 * x[1] + 80 * x[2]
        ),
    ]
    constraints = [x >= 0, x <= 10, cp.sum_squares(x) <= 1]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve(solver="ADENA")
    assert pb.status == "max_pbs_solved_reached"
    assert objective_values.shape == (353, 3)
    assert x.values.shape == (353, 3)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(objective_values[:, 2] == objectives[2].values)
    x_values = x.values
    assert np.all(
        objective_values[:, 0]
        == x_values[:, 0] ** 2
        + x_values[:, 1] ** 2
        + x_values[:, 2] ** 2
        + 10 * x_values[:, 1]
        - 120 * x_values[:, 2]
    )
    assert np.all(
        objective_values[:, 1]
        == x_values[:, 0] ** 2
        + x_values[:, 1] ** 2
        + x_values[:, 2] ** 2
        + 80 * x_values[:, 0]
        - 448 * x_values[:, 1]
        + 80 * x_values[:, 2]
    )
    assert np.all(
        objective_values[:, 2]
        == x_values[:, 0] ** 2
        + x_values[:, 1] ** 2
        + x_values[:, 2] ** 2
        - 448 * x_values[:, 0]
        + 80 * x_values[:, 1]
        + 80 * x_values[:, 2]
    )
