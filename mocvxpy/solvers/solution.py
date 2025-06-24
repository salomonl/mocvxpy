import cdd
import numpy as np

from cdd import gmp
from fractions import Fraction
from mocvxpy.constants import MIN_DIST_OBJ_VECS, MIN_TOL_HYPERPLANES
from typing import Optional, Sequence


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
