import numpy as np


def update_local_lower_bounds(
    lower_bounds: np.ndarray, y: np.ndarray, nobj: int
) -> np.ndarray:
    """Update a local lower bounds set.

    Given an objective vector, update a set of local lower bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 2 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsibility of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    lower_bounds: np.ndarray
        The set of local lower bounds. Has dimensions |L| x nobj.
    y: np.ndarray
        An update point of dimension nobj.
    nobj: int
        The number of objectives of the problem.

    Returns
    -------
    list[np.ndarray]
       The set of local lower bounds updated.
    """
    # Compute A = {l in lower_bounds | y > l}: Search zones that contain y
    L = lower_bounds
    A_indexes = np.all(np.greater(y, L), axis=1)

    # Compute Bi = {l in lower_bounds | yi = li and y(_i) > l(_i)} for i = 1, 2, ..., nobj
    # where y(_i) = (y1, ..., yi-1, y(i+1), ..., ym)
    B = [None] * nobj
    for i in range(nobj):
        B[i] = np.all(
            np.vstack(
                [
                    np.equal(y[i], L[:, i]),
                    np.greater(y[0:i], L[:, 0:i]).T,
                    np.greater(y[i + 1 :], L[:, i + 1 :]).T,
                ]
            ),
            axis=0,
        )

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local lower bounds of A
    P = [L[A_indexes] for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {l in Pi | l not >= l' for all l' in Pi cup Bi, l' != l}
    # for i = 1, 2, ..., m
    nlower_bounds = L.shape[0]
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        # Remove all dominated points from Pi by the ones of Pi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominated
                if comparison_status == 1:
                    P_indexes[i][ind1] = False
                    break
                # Point pind1 dominates or is equal to point pind2
                elif comparison_status == 2 or comparison_status == 3:
                    P_indexes[i][ind2] = False

            # no need to keep on iterating, the point pind1 is dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1 + 1, P[i].shape[0]):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominated
                if comparison_status == 1:
                    P_indexes[i][ind1] = False
                    break
                # Point pind1 dominates or is equal to point pind2
                elif comparison_status == 2 or comparison_status == 3:
                    P_indexes[i][ind2] = False

        # Remove points of Pi dominated by points of Bi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(nlower_bounds):
                if not B[i][ind2]:
                    continue

                comparison_status = compare_objective_vectors(P[i][ind1], L[ind2])
                # Point at index ind1 is dominated by another point of Bi or equal
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    new_lower_bounds = np.vstack([L[~A_indexes, :], np.vstack(P)])

    return new_lower_bounds


def local_lower_bounds(
    objective_pts: np.ndarray, llb_extreme_eps: float = 1e-4
) -> np.ndarray:
    """Compute a local lower bound set.

    Given a set of objective vectors, compute a local lower bound set
    of a multiobjective problem.

    NB: it is the user's responsability to ensure that the set of
    objective vectors is stable, i.e., all its elements are different
    and non-dominated between them.

    Arguments
    ---------
    objective_pts: np.ndarray
        The set of non-dominated objective points. Has dimensions (|N|, nobj).
    llb_extreme_eps: float
        A positive tolerance value that controls the distance of extreme
        local lower bound elements to extreme objective_pts.

    Returns
    -------
    np.ndarray
       The local lower bound set. Has dimensions (|LLB|, nobj).
    """
    if llb_extreme_eps <= 0.0:
        raise ValueError("llb_extreme_eps must be positive", llb_extreme_eps)

    if objective_pts.ndim != 2:
        raise ValueError(
            "The objective points is not given as a matrix", objective_pts.ndim
        )

    nobj = objective_pts.shape[1]
    if nobj <= 1:
        raise ValueError(
            "The number of objectives must be superior or equal to 2", nobj
        )
    nobj_pts = objective_pts.shape[0]
    if nobj_pts < 1:
        raise ValueError("The number of objective points must be positive", nobj_pts)

    llb = np.min(objective_pts, axis=0) - llb_extreme_eps
    llb = np.reshape(llb, (1, nobj))
    for y in objective_pts:
        llb = update_local_lower_bounds(llb, y, nobj)

    return llb


def update_local_upper_bounds(
    upper_bounds: np.ndarray, y: np.ndarray, nobj: int
) -> np.ndarray:
    """Update a local upper bounds set.

    Given an objective vector, update a set of local upper bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 1 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsibility of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    upper_bounds: np.ndarray
        The set of local upper bounds. Has dimensions (|U|, nobj).
    y: np.ndarray
        An update point of dimension nobj.
    nobj: int
        The number of objectives of the problem.

    Returns
    -------
    list[np.ndarray]
       The set of local upper bounds updated.
    """
    # Compute A = {u in upper_bounds | y < u}: Search zones that contain y
    # U = np.array(upper_bounds)
    U = upper_bounds
    A_indexes = np.all(np.less(y, U), axis=1)

    # Compute Bi = {u in upper_bounds | yi = ui and y(_i) < u(_i)} for i = 1, 2, ..., nobj
    # where u(_i) = (y1, ..., yi-1, u(i+1), ..., ym)
    B = [None] * nobj
    for i in range(nobj):
        B[i] = np.all(
            np.vstack(
                [
                    np.equal(y[i], U[:, i]),
                    np.less(y[0:i], U[:, 0:i]).T,
                    np.less(y[i + 1 :], U[:, i + 1 :]).T,
                ]
            ),
            axis=0,
        )

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local upper bounds of A
    P = [U[A_indexes] for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {u in Pi | u not <= u' for all u' in Pi cup Bi, u' != u}
    # for i = 1, 2, ..., m
    nupper_bounds = U.shape[0]
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        for ind1 in range(P[i].shape[0]):
            # The point has been already processed
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominating
                if comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break
                # Point pind2 dominates or equals point pind1
                elif comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind2] = False

            # no need to continue, the point pind1 is dominating
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1 + 1, P[i].shape[0]):
                comparison_status = compare_objective_vectors(P[i][ind1], P[i][ind2])
                # Point at index ind1 is dominating
                if comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break
                # Point pind2 dominates or equals point pind1
                elif comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind2] = False

        # Filter out redundant points of Pi by Bi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominating
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(nupper_bounds):
                if not B[i][ind2]:
                    continue

                comparison_status = compare_objective_vectors(P[i][ind1], U[ind2])
                # Point pind1 dominates or equals a point in B
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    # new_upper_bounds = (upper_bounds \ A) cup P
    new_upper_bounds = np.vstack([U[~A_indexes, :], np.vstack(P)])

    return new_upper_bounds


def local_upper_bounds(
    objective_pts: np.ndarray, lub_extreme_eps: float = 1e-4
) -> np.ndarray:
    """Compute a local upper bound set.

    Given a set of objective vectors, compute a local upper bound set
    of a multiobjective problem.

    NB: it is the user's responsability to ensure that the set of
    objective vectors is stable, i.e., all its elements are different
    and non-dominated between them.

    Arguments
    ---------
    objective_pts: np.ndarray
        The set of non-dominated objective points. Has dimensions (|N|, nobj).
    lub_extreme_eps: float
        A positive tolerance value that controls the distance of extreme
        local upper bound elements to extreme objective_pts.

    Returns
    -------
    np.ndarray
       The local upper bound set. Has dimensions (|LUB|, nobj).
    """
    if lub_extreme_eps <= 0.0:
        raise ValueError("lub_extreme_eps must be positive", lub_extreme_eps)

    if objective_pts.ndim != 2:
        raise ValueError(
            "The objective points is not given as a matrix", objective_pts.ndim
        )

    nobj = objective_pts.shape[1]
    if nobj <= 1:
        raise ValueError(
            "The number of objectives must be superior or equal to 2", nobj
        )
    nobj_pts = objective_pts.shape[0]
    if nobj_pts < 1:
        raise ValueError("The number of objective points must be positive", nobj_pts)

    lub = np.max(objective_pts, axis=0) + lub_extreme_eps
    lub = np.reshape(lub, (1, nobj))
    for y in objective_pts:
        lub = update_local_upper_bounds(lub, y, nobj)

    return lub


def compare_objective_vectors(y1: np.ndarray, y2: np.ndarray) -> int:
    """Compare two objective vectors according to Pareto dominance.

    Arguments
    ---------
    y1: np.ndarray
        The first objective vector.
    y2: np.ndarray
        The second objective vector.

    Returns
    -------
    int:
       0 if y1 is non-dominated by y2;
       1 if y1 is dominated by y2;
       2 if y1 is dominating y2;
       3 if y1 and y2 are equal.
    """
    is_better = np.any(np.less(y1, y2))
    is_worse = np.any(np.less(y2, y1))
    if is_worse:
        if is_better:
            # non-dominated
            return 0
        else:
            # dominated
            return 1
    else:
        if is_better:
            # dominating
            return 2
        else:
            # equal
            return 3
