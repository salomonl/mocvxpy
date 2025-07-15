import cdd
import numpy as np

from cdd import gmp
from fractions import Fraction
from mocvxpy.constants import MIN_DIST_OBJ_VECS, MIN_TOL_HYPERPLANES
from typing import List, Optional, Sequence


class Solution:
    """Contains the solutions of a multiobjective problem.

    Attributes:
    ----------
    nvars : int
        The number of decision variables of the problem
    nobj : int
        The number of objective of the problem
    """

    def __init__(self, nvars: int, nobj: int) -> None:
        if nvars <= 0:
            raise ValueError(
                "The number of variables of a problem must be positive", nvars
            )
        if nobj <= 1:
            raise ValueError(
                "The number of objectives of a problem must be superior or equal to 2",
                nobj,
            )

        self._nvars = nvars
        self._nobj = nobj
        self._xvalues = np.array([], dtype=np.float64).reshape((0, nvars))
        self._objective_values = np.array([], dtype=np.float64).reshape((0, nobj))

    @property
    def xvalues(self) -> np.ndarray:
        """Accessor method for decision variables values.

        Returns
        -------
        np.ndarray
           A matrix of decision variables values of dimensions |X| x nvars
        """
        return self._xvalues

    @property
    def objective_values(self) -> np.ndarray:
        """Accessor method for objective values.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions |X| x nobj
        """
        return self._objective_values

    def insert_solution(self, x: np.ndarray, fvalues: np.ndarray) -> None:
        """Insert solution into the Solution set.

        Arguments
        -------
        x: np.ndarray
           A decision vector: it is the responsability from the user to check
           it is an efficient solution

        fvalues: np.ndarray
           A objective vector: it is the responsability from the user to check
           it is non dominated
        """
        if x.shape != (self._nvars,):
            raise ValueError(
                "The dimensions of x are not compatible with the other solutions",
                x.shape,
            )
        if fvalues.shape != (self._nobj,):
            raise ValueError(
                "The dimensions of fvalues are not compatible with the other objective vectors",
                fvalues.shape,
            )

        self._xvalues = np.vstack([self._xvalues, x])
        self._objective_values = np.vstack([self._objective_values, fvalues])

    def compute_hrepresentation_from_objective_values(
        self,
    ) -> Optional[Sequence[Sequence[float]]]:
        """Compute h-representation from the set of objective vectors.

        Returns
        -------
        None
            if the set of objective values is empty.

        List[List[float]]
            a list of hyperplanes that characterize the h representation of the objective set.
            An hyperplane 0 <= b + a^T y is represented by the following list:
            [b, a1, ..., am]
            where m is the number of objectives
        """
        if self.objective_values.shape[0] == 0:
            return

        v_representation = [[1.0] + vertex for vertex in self.objective_values]
        # Complete by rays
        nobj = self._nobj
        for obj in range(nobj):
            ray = [0.0 for _ in range(nobj + 1)]
            ray[obj + 1] = 1.0
            v_representation.append(ray)

        try:
            vmat = cdd.matrix_from_array(
                v_representation, rep_type=cdd.RepType.GENERATOR
            )
            cdd.matrix_canonicalize(vmat)
            vpoly = cdd.polyhedron_from_matrix(vmat)
            hmat = cdd.copy_inequalities(vpoly)
            cdd.matrix_redundancy_remove(hmat)
            return hmat.array
        except RuntimeError:
            # Recompute but use the exact precision
            v_exact_representation = []
            for vertex in v_representation:
                v_exact = [Fraction(vi) for vi in vertex]
                v_exact_representation.append(v_exact)
            vmat = gmp.matrix_from_array(
                v_exact_representation, rep_type=cdd.RepType.GENERATOR
            )
            gmp.matrix_canonicalize(vmat)
            vpoly = gmp.polyhedron_from_matrix(vmat)
            hmat_exact = gmp.copy_inequalities(vpoly)
            hmat = []
            for halfspace in hmat_exact.array:
                hmat.append([float(hi) for hi in halfspace])
            return hmat

    def extreme_objective_vectors(self) -> np.ndarray:
        """Compute the set of extreme objective vectors.

        An extreme objective vector is a point in the objective space
        who reaches a minimum f value for one of the solutions.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions nobj x nobj:
           row i corresponds to objective i.
        """
        extreme_pts_indexes = np.argmin(self.objective_values, axis=0)
        return self.objective_values[extreme_pts_indexes, :]

    def ideal_objective_vector(self) -> np.ndarray:
        """Compute the set of extreme objective vectors.

        An extreme objective vector is a point in the objective space
        who reaches a minimum f value for one of the solutions.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions nobj x nobj:
           row i corresponds to objective i.
        """
        return np.min(self.objective_values, axis=0)


class OuterApproximation:
    """Defines an outer approximation of a solution set
    for a multiobjective problem.

    Attributes:
    ----------
    nobj : int
        The number of objectives of the problem
    """

    def __init__(self, nobj: int) -> None:
        if nobj <= 1:
            raise ValueError(
                "The number of objectives of a problem must be superior or equal to 2",
                nobj,
            )
        self._halfspaces = np.array([], dtype=np.float64).reshape((0, nobj + 1))
        self._dim = nobj

    @property
    def halfspaces(self) -> np.ndarray:
        """Returns the halfspaces that define the outer approximation."""
        return self._halfspaces

    def insert_halfspace(self, halfspace: np.ndarray) -> None:
        """Insert a new halfspace into the current outer approximation

        Attributes:
        ----------
        halfspace : np.ndarray
           A vector [beta, a1, a2, ..., am] that characterizes the hyperplane
           0 <= beta + a^T
        """
        if halfspace.size != self._dim + 1:
            return

        self._halfspaces = np.vstack([self._halfspaces, halfspace])

    def compute_vertices(self) -> Optional[np.ndarray]:
        """Compute the vertices of the outer approximation."""
        if self.halfspaces.size == 0:
            return None

        # Try removing redundant hyperplanes
        try:
            outer_hrep = cdd.matrix_from_array(
                self._halfspaces.tolist(), rep_type=cdd.RepType.INEQUALITY
            )
            cdd.matrix_canonicalize(outer_hrep)
            self._halfspaces = []
            for halfspace in outer_hrep.array:
                # Apply some rounding to try to mitigate numerical errors
                self._halfspaces.append(
                    [h if abs(h) >= MIN_TOL_HYPERPLANES else 0.0 for h in halfspace]
                )
            self._halfspaces = np.asarray(self._halfspaces)
        except RuntimeError:
            pass

        # Generate outer vertices
        outer_hrep_exact = []
        for halfspace in self._halfspaces:
            halfspace_exact = [Fraction(h) for h in halfspace]
            outer_hrep_exact.append(halfspace_exact)
        outer_hrep_exact = gmp.matrix_from_array(
            outer_hrep_exact, rep_type=cdd.RepType.INEQUALITY
        )
        poly = gmp.polyhedron_from_matrix(outer_hrep_exact)
        outer_vrep = gmp.copy_generators(poly)

        # NB: the V-representation returned by cddlib looks like:
        # First column: 1 if it is a vertex, 0 if it is a ray; other columns are the coordinates of the vertices/rays
        outer_vertices = [
            [float(vi) for vi in vertex[1:]]
            for vertex in outer_vrep.array
            if vertex[0] == 1.0
        ]
        # In the case where no vertex is present, it means that the polyhedron does not possess any vertex or
        # has only (0, ..., 0) as a vertex, which will supposed to be the case for the moment
        if not outer_vertices:
            outer_vertices.append([0.0 for _ in range(self._dim)])

        # Remove all vertices that are too close to each other
        # Taken from D. Dorfler original implementation of this algorithm
        indices = []
        for ind_i in range(len(outer_vertices)):
            for ind_j in range(ind_i + 1, len(outer_vertices)):
                if (
                    np.linalg.norm(
                        np.asarray(outer_vertices[ind_j])
                        - np.asarray(outer_vertices[ind_i])
                    )
                    <= MIN_DIST_OBJ_VECS
                ):
                    indices.append(ind_i)
                    break
        for ind in sorted(indices, reverse=True):
            del outer_vertices[ind]

        return np.asarray(outer_vertices)


def update_local_lower_bounds(
    lower_bounds: List[np.ndarray], y: np.ndarray, nobj: int
) -> List[np.ndarray]:
    """Update a local lower bounds set.

    Given an objective vector, update a set of local lower bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 2 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsability of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    lower_bounds: list[np.ndarray]
        The set of local lower bounds. Each objective vector is of dimension nobj
    y: np.ndarray
        An update point of dimension nobj.

    Returns
    -------
    list[np.ndarray]
       The set of local lower bounds updated.
    """
    # Compute A = {l in lower_bounds | y > l}: Search zones that contain y
    L = np.array(lower_bounds)
    nlower_bounds = L.shape[0]
    A_indexes = np.zeros(nlower_bounds, dtype=bool)
    for ind in range(nlower_bounds):
        if (y > L[ind]).all():
            A_indexes[ind] = True

    # Compute Bi = {l in lower_bounds | yi = li and y(_i) > l(_i)} for i = 1, 2, ..., nobj
    # where y(_i) = (y1, ..., yi-1, y(i+1), ..., ym)
    B = [np.zeros(nlower_bounds, dtype=bool) for _ in range(nobj)]
    for i in range(nobj):
        for ind in range(nlower_bounds):
            if y[i] != L[ind][i]:
                continue
            append_to_Bi = (y[0:i] > L[ind, 0:i]).all()
            if not append_to_Bi:
                continue
            append_to_Bi = (y[i + 1 :] > L[ind, i + 1 :]).all()
            B[i][ind] = append_to_Bi

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local lower bounds of A
    P = [np.copy(L[A_indexes]) for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {l in Pi | l not >= l' for all l' in Pi cup Bi, l' != l}
    # for i = 1, 2, ..., m
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        # Remove all dominated points from Pi by the ones of Pi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(
                    P[i][ind1], P[i][ind2], nobj
                )
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
                comparison_status = compare_objective_vectors(
                    P[i][ind1], P[i][ind2], nobj
                )
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

                comparison_status = compare_objective_vectors(P[i][ind1], L[ind2], nobj)
                # Point at index ind1 is dominated by another point of Bi or equal
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    # new_lower_bounds = (lower_bounds \ A) cup P
    new_lower_bounds = []
    for l, is_in_A in zip(lower_bounds, A_indexes.tolist()):
        if not is_in_A:
            new_lower_bounds.append(np.copy(l))
    for i in range(nobj):
        for ind in range(P[i].shape[0]):
            new_lower_bounds.append(np.copy(P[i][ind]))

    return new_lower_bounds


def update_local_upper_bounds(
    upper_bounds: List[np.ndarray], y: np.ndarray, nobj: int
) -> List[np.ndarray]:
    """Update a local upper bounds set.

    Given an objective vector, update a set of local upper bounds of a
    multiobjective problem.

    The implementation follows the description of Algorithm 1 given in:
    Eichfelder, G., & Warnow, L. (2023).
    "Advancements in the computation of enclosures for multi-objective optimization problems."
    European Journal of Operational Research, 310(1), p. 315-327.
    https://doi.org/10.1016/j.ejor.2023.02.032

    The algorithm is not implemented in a Python class to be the most efficient possible.
    It is the responsability of the user to assure the arguments satisfy the correct assumptions.

    Arguments
    ---------
    upper_bounds: list[np.ndarray]
        The set of local upper bounds. Each objective vector is of dimension nobj
    y: np.ndarray
        An update point of dimension nobj.

    Returns
    -------
    list[np.ndarray]
       The set of local upper bounds updated.
    """
    # Compute A = {u in upper_bounds | y < u}: Search zones that contain y
    U = np.array(upper_bounds)
    nupper_bounds = U.shape[0]
    A_indexes = np.zeros(nupper_bounds, dtype=bool)
    for ind in range(nupper_bounds):
        if (y < U[ind]).all():
            A_indexes[ind] = True

    # Compute Bi = {u in upper_bounds | yi = ui and y(_i) < u(_i)} for i = 1, 2, ..., nobj
    # where u(_i) = (y1, ..., yi-1, u(i+1), ..., ym)
    B = [np.zeros(nupper_bounds, dtype=bool) for _ in range(nobj)]
    for i in range(nobj):
        for ind in range(nupper_bounds):
            if y[i] != U[ind][i]:
                continue
            append_to_Bi = (y[0:i] < U[ind, 0:i]).all()
            if not append_to_Bi:
                continue
            append_to_Bi = (y[i + 1 :] < U[ind, i + 1 :]).all()
            B[i][ind] = append_to_Bi

    # Compute Pi for i = 1, 2, ..., nobj: generate all projections of y on the local upper bounds of A
    P = [np.copy(U[A_indexes]) for _ in range(nobj)]
    for i in range(nobj):
        P[i][:, i] = y[i]

    # Filter out all redundant points of P: Pi = {u in Pi | u not <= u' for all u' in Pi cup Bi, u' != u}
    # for i = 1, 2, ..., m
    P_indexes = [np.ones(P[i].shape[0], dtype=bool) for i in range(nobj)]
    for i in range(nobj):
        for ind1 in range(P[i].shape[0]):
            # The point has been already processed
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(ind1):
                comparison_status = compare_objective_vectors(
                    P[i][ind1], P[i][ind2], nobj
                )
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
                comparison_status = compare_objective_vectors(
                    P[i][ind1], P[i][ind2], nobj
                )
                # Point at index ind1 is dominating
                if comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break
                # Point pind2 dominates or equals point pind1
                elif comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind2] = False

        # Filter out redundant points of Pi by Bi
        for ind1 in range(P[i].shape[0]):
            # The point is already dominated
            if not P_indexes[i][ind1]:
                continue

            for ind2 in range(nupper_bounds):
                if not B[i][ind2]:
                    continue

                comparison_status = compare_objective_vectors(P[i][ind1], U[ind2], nobj)
                # Point pind1 dominates or equals a point in B
                if comparison_status == 1 or comparison_status == 3:
                    P_indexes[i][ind1] = False
                    break

    for i in range(nobj):
        P[i] = P[i][P_indexes[i]]

    # new_upper_bounds = (upper_bounds \ A) cup P
    new_upper_bounds = []
    for l, is_in_A in zip(upper_bounds, A_indexes.tolist()):
        if not is_in_A:
            new_upper_bounds.append(np.copy(l))
    for i in range(nobj):
        for ind in range(P[i].shape[0]):
            new_upper_bounds.append(np.copy(P[i][ind]))

    return new_upper_bounds


def compare_objective_vectors(y1: np.ndarray, y2: np.ndarray, nobj: int) -> int:
    """Compare two objective vectors according to Pareto dominance.

    Arguments
    ---------
    y1: np.ndarray
        The first objective vector.
    y2: np.ndarray
        The second objective vector.
    nobj: int
        The number of objectives.

    Returns
    -------
    int:
       0 if y1 is non-dominated by y2;
       1 if y1 is dominated by y2;
       2 if y1 is dominating y2;
       3 if y1 and y2 are equal.
    """
    is_better = False
    is_worse = False
    for obj in range(nobj):
        if y1[obj] < y2[obj]:
            is_better = True
        if y2[obj] < y1[obj]:
            is_worse = True
        if is_worse and is_better:
            break
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
