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
import sys

from matplotlib import pyplot as plt

# Application inspired by the following article:
#
# Davoodi, E., Babaei, E., Mohammadi-Ivatloo, B., Shafie-Khah, M., & Catalão, J. P. (2020).
# Multiobjective optimal power flow using a semidefinite programming-based model.
# IEEE systems journal, 15(1), 158-169.
#
# The problem formulation is taken from:
#
# Lupien, J. L., & Lesage-Landry, A. (2025).
# Ex post conditions for the exactness of optimal power flow conic
# relaxations.
# Electric Power Systems Research, 238, 111130.
#
# The code for data processing is strongly inspired by:
# https://github.com/LORER-MTL/OPF_Tools/

if __name__ == "__main__":
    try:
        import mosek
    except ImportError:
        print("Please install Mosek to run this example.")
        sys.exit(1)

    # case 9 from matpower
    # system baseMVA
    base_mva = 100

    # bus data
    # bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin
    bus_data = np.array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [2, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [3, 2, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [4, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [5, 1, 90, 30, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [6, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [7, 1, 100, 35, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [8, 1, 0, 0, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
            [9, 1, 125, 50, 0, 0, 1, 1, 0, 345, 1, 1.1, 0.9],
        ]
    )

    # generator data
    # bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin    Pc1    Pc2    Qc1min    Qc1max    Qc2min    Qc2max    ramp_agc    ramp_10    ramp_30    ramp_q    apf
    gen_data = np.array(
        [
            [1, 0, 0, 300, -300, 1, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 85, 0, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # branch data
    # fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax
    branch_data = np.array(
        [
            [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1, -360, 360],
            [4, 5, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
            [5, 6, 0.039, 0.17, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
            [3, 6, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1, -360, 360],
            [6, 7, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
            [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
            [8, 2, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1, -360, 360],
            [8, 9, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
            [9, 4, 0.01, 0.085, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
        ]
    )

    # Generator cost data
    # 1    startup    shutdown    n    x1    y1    ...    xn    yn
    # 2    startup    shutdown    n    c(n-1)    ...    c0
    generator_costs = np.array(
        [
            [2, 1500, 0, 3, 0.11, 5, 150],
            [2, 2000, 0, 3, 0.085, 1.2, 600],
            [2, 3000, 0, 3, 0.1225, 1, 335],
        ]
    )

    # Compute the admittance matrix Y
    number_of_buses = bus_data.shape[0]
    Y = np.zeros((number_of_buses, number_of_buses), dtype=complex)
    for branch_logs in branch_data:
        i, j = int(branch_logs[0]) - 1, int(branch_logs[1]) - 1
        impedance_r, impedance_x = branch_logs[2], branch_logs[3]
        Y_val = 1 / (impedance_r + 1.0j * impedance_x)
        Y[i, j] = Y_val
        Y[j, i] = Y_val

    # Compute the apparent power limit matrix Smax
    Smax = np.zeros((number_of_buses, number_of_buses))
    for branch_logs in branch_data:
        i, j = int(branch_logs[0]) - 1, int(branch_logs[1]) - 1
        Smax_val = branch_logs[5] / base_mva
        Smax[i, j] = Smax_val
        Smax[j, i] = Smax_val

    # Compute bus load data matrix:
    # - the first column contains the active power Pd to load at each bus
    # - the second column contains the reactive power Qd to load at each bus
    bus_load_data = np.zeros((number_of_buses, 2))
    for bus_logs in bus_data:
        bus_ind = int(bus_logs[0]) - 1
        bus_load_data[bus_ind, 0] = bus_logs[2] / base_mva
        bus_load_data[bus_ind, 1] = bus_logs[3] / base_mva

    # Get voltage limit
    # - the first column contains Vlim_max.
    # - the second column contains Vlim_min.
    Vlim = np.zeros((number_of_buses, 2))
    for bus_logs in bus_data:
        bus_ind = int(bus_logs[0]) - 1
        Vlim[bus_ind, 0] = bus_logs[11]
        Vlim[bus_ind, 1] = bus_logs[12]

    # Get generator costs: for each generator, the cost is
    # given as f(Pg) = c2 * Pg**2 + c1 * Pg + c0
    cost_matrix = np.zeros((number_of_buses, 3))
    for ind_gen, gen_cost_logs in enumerate(generator_costs):
        cost_matrix[ind_gen, 0] = gen_cost_logs[4]
        cost_matrix[ind_gen, 1] = gen_cost_logs[5]
        cost_matrix[ind_gen, 2] = gen_cost_logs[6]

    # Get lines
    lines = []
    for branch_logs in branch_data:
        i, j = int(branch_logs[0]) - 1, int(branch_logs[1]) - 1
        lines.append((i, j))

    # Define the problem: use a SDR relaxation to solve it
    W = mocp.Variable((number_of_buses, number_of_buses), hermitian=True)

    # Power transfer variables
    pij = mocp.Variable(len(lines))
    pji = mocp.Variable(len(lines))
    qij = mocp.Variable(len(lines))
    qji = mocp.Variable(len(lines))

    # Active power of generator g in G
    Pg = mocp.Variable(number_of_buses)
    # Reactive power of generator
    Qg = mocp.Variable(number_of_buses)

    # Define the constraints
    constraints = []

    for bus in range(number_of_buses):

        # For all i in buses, sum pij = pi and sum qij = qi
        sum_p = 0.0
        sum_q = 0.0
        for ind_line, line in enumerate(lines):
            bus_start, bus_end = line[0], line[1]
            if bus == bus_start:
                sum_p += pij[ind_line]
                sum_q += qij[ind_line]
            elif bus == bus_end:
                sum_p += pji[ind_line]
                sum_q += qji[ind_line]

        constraints.append(sum_p == Pg[bus] - bus_load_data[bus, 0])
        constraints.append(sum_q == Qg[bus] - bus_load_data[bus, 1])

        # Voltage limits
        constraints += [
            cp.real(W[bus, bus]) >= Vlim[bus, 1] ** 2,
            cp.real(W[bus, bus]) <= Vlim[bus, 0] ** 2,
        ]

    # Bounds on active and reactive generator powers
    # When there are none, Pg[i] and Qg[i] are set to 0
    generator_bus_id = []
    for gen_logs in gen_data:
        bus_id = gen_logs[0] - 1
        Pmax, Pmin = gen_logs[8] / base_mva, gen_logs[9] / base_mva
        Qmax, Qmin = gen_logs[3] / base_mva, gen_logs[4] / base_mva
        constraints += [Pg[bus_id] <= Pmax, Pg[bus_id] >= Pmin]
        constraints += [Qg[bus_id] <= Qmax, Qg[bus_id] >= Qmin]
        generator_bus_id.append(bus_id)

    for bus in range(number_of_buses):
        if bus in generator_bus_id:
            continue

        constraints += [Pg[bus] == 0]
        constraints += [Qg[bus] == 0]

    # Power flow equations
    for ind_line, line in enumerate(lines):
        i, j = line[0], line[1]
        constraints += [
            pij[ind_line] + 1.0j * qij[ind_line]
            == (W[i, i] - W[i, j]) * cp.conj(Y[i, j])
        ]
        constraints += [
            pji[ind_line] + 1.0j * qji[ind_line]
            == (W[j, j] - W[j, i]) * cp.conj(Y[j, i])
        ]

        # Apparent power
        if not Smax[i, j] == 0:
            constraints += [
                cp.square(pij[ind_line]) + cp.square(qij[ind_line])
                <= cp.square(Smax[i, j])
            ]
            constraints += [
                cp.square(pji[ind_line]) + cp.square(qji[ind_line])
                <= cp.square(Smax[j, i])
            ]

    constraints += [W >> 0]

    # Define objectives
    # f1 = sum fg(Pg)
    costs = 0
    for bus_id, cost_info in enumerate(cost_matrix):
        c2 = cost_info[0]
        c1 = cost_info[1]
        c0 = cost_info[2]
        if c1 > 0:
            costs += (
                c0 + c1 * Pg[bus_id] * base_mva + c2 * cp.square(Pg[bus_id] * base_mva)
            )

    # Minimize power losses
    # P_loss = Re(sum_{b in bus} vb ib*) = Re(sum_{(i, j) in line} (Wii - Wij) Yij*)
    power_losses = 0
    for line in lines:
        i, j = line[0], line[1]
        power_losses += cp.real((W[i, i] - W[i, j]) * cp.conj(Y[i, j]))
        power_losses += cp.real((W[j, j] - W[j, i]) * cp.conj(Y[j, i]))

    objectives = [cp.Minimize(costs), cp.Minimize(power_losses)]
    pb = mocp.Problem(objectives, constraints)

    # Solve the relaxation problem
    # Do not work with other SDP solvers
    objective_values = pb.solve(
        solver="ADENA",
        stopping_tol=1e-4,
        scalarization_solver_options={"solver": cp.MOSEK},
        verbose=True,
    )

    # Obtain the network policies that are exact, as described in
    #
    # Lupien, J. L., & Lesage-Landry, A. (2025).
    # Ex post conditions for the exactness of optimal power flow conic
    # relaxations.
    # Electric Power Systems Research, 238, 111130.
    #
    TOL_RANK = 1e-6
    TOL_COND1 = 1e-4
    TOL_COND2 = 5 * 1e-3
    cycle_buses_id = [4, 5, 6, 7, 8, 9]

    exact_policy_indexes = []
    for ind_W, W_vals in enumerate(W.values):
        # 1- Check if W is approximately of rank 1, using SVD
        _, S, _ = np.linalg.svd(W_vals)
        second_eigval = S[1]

        # NB: the eigenvalues are ordered by decreasing values
        if np.abs(second_eigval) > TOL_RANK:
            continue

        # 2- For all ij in lines, Wii Wjj = |Wij|^2
        cond1_is_satisfied = True
        for line in lines:
            i, j = line[0], line[1]
            if (
                abs(
                    W_vals[i, i] * W_vals[j, j]
                    - (np.real(W_vals[i, j]) ** 2 + np.imag(W_vals[i, j]) ** 2)
                )
                > TOL_COND1
            ):
                cond1_is_satisfied = False
                break

        if not cond1_is_satisfied:
            continue

        # 3- For every cycle c in C for the electric graph
        # (buses, lines), Im(sum_{(i, j) in c} Wij) = 0 mod 2pi
        im_C_val = 0.0
        for bus_id in range(len(cycle_buses_id)):
            i, j = (
                cycle_buses_id[bus_id] - 1,
                cycle_buses_id[(bus_id + 1) % len(cycle_buses_id)] - 1,
            )
            im_C_val += np.imag(W_vals[i, j])

        cond2_is_satisfied = (
            abs((im_C_val / 2 * np.pi) - int(im_C_val / 2 * np.pi)) <= TOL_COND2
        )
        if not cond2_is_satisfied:
            continue

        exact_policy_indexes.append(ind_W)

    print("Number of exact energy policies: ", len(exact_policy_indexes))

    # Plot the exact and relaxed policies
    opt_power_costs = objective_values[:, 0]
    opt_power_losses = objective_values[:, 1]

    ax = plt.figure().add_subplot()
    ax.scatter(
        opt_power_costs,
        opt_power_losses,
        facecolor="none",
        edgecolors="blue",
        alpha=0.7,
        label="Pareto-Optimal relaxed policy",
    )
    ax.scatter(
        opt_power_costs[exact_policy_indexes],
        opt_power_losses[exact_policy_indexes],
        marker="x",
        s=80,
        color="red",
        label="Pareto optimal exact policy",
    )
    plt.legend()
    plt.xlabel("Power costs ($)")
    plt.ylabel("Power losses (MW)")

    plt.show()
