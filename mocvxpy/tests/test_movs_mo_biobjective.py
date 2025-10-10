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
import mocvxpy as mocp
import numpy as np


def test_solve_circle_pb_with_MOVS():
    x = mocp.Variable(2)
    objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
    constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve()
    assert pb.status == "scalarization_pb_numeric"
    assert objective_values.shape == (33, 2)
    assert x.values.shape == (33, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    assert np.all(x.values == objective_values)


def test_solve_norm1_min_st_qp_constraints_with_MOVS():
    x = mocp.Variable(2)
    objectives = [
        cp.Minimize(cp.norm(x, 1)),
        cp.Minimize(cp.norm(x - np.array([2, 0]), 1)),
    ]
    constraints = [cp.sum_squares(x) <= 100]
    pb = mocp.Problem(objectives, constraints)

    objective_values = pb.solve()
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

    objective_values = pb.solve()
    assert pb.status == "optimal"
    assert objective_values.shape == (101, 2)
    assert x.values.shape == (101, 2)
    assert np.all(objective_values[:, 0] == objectives[0].values)
    assert np.all(objective_values[:, 1] == objectives[1].values)
    x_values = x.values
    assert np.all(
        objective_values[:, 0] == np.sum((x_values - np.array([3.0, 3.0])) ** 2, axis=1)
    )
    assert np.all(
        objective_values[:, 1] == np.sum((x_values - np.array([1.0, 1.0])) ** 2, axis=1)
    )
