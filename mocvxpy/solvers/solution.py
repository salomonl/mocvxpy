import numpy as np

from mocvxpy.constants import MIN_DIST_OBJ_VECS
from mocvxpy.utilities.polyhedron import Polyhedron
from typing import Optional


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
        self._dual_objective_values = np.array([], dtype=np.float64).reshape((0, nobj))
        self._dual_constraint_values = None

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

    @property
    def dual_objective_values(self) -> np.ndarray:
        """Accessor method for dual objective values.

        Returns
        -------
        np.ndarray
           A matrix of dual objective values (weights) of dimensions |X| x nobj
        """
        return self._dual_objective_values

    @property
    def dual_constraint_values(self) -> Optional[np.ndarray]:
        """Accessor method for dual objective values.

        Returns
        -------
        Optional[np.ndarray]
           A matrix of dual constraint values (weights) of dimensions |X| x cdims
        """
        return self._dual_constraint_values

    def insert_solution(
        self,
        x: np.ndarray,
        fvalues: np.ndarray,
        dual_fvalues: np.ndarray,
        dual_values: Optional[np.ndarray] = None,
    ) -> None:
        """Insert solution into the Solution set.

        Arguments
        -------
        x: np.ndarray
           A decision vector: it is the responsability from the user to check
           it is an efficient solution

        fvalues: np.ndarray
           A objective vector: it is the responsability from the user to check
           it is non dominated

        dual_fvalues: np.ndarray
           The corresponding dual objective values.

        dual_values: Optional[np.ndarray]
           The dual constraint values.
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
        if dual_fvalues.shape != (self._nobj,):
            raise ValueError(
                "The dimensions of dual_fvalues are not compatible with the other dual objective vectors",
                dual_fvalues.shape,
            )
        if dual_values is None:
            if self._dual_constraint_values is not None:
                raise ValueError("Dual values for constraints must be provided")
        else:
            if self._dual_constraint_values is not None:
                ncons = self._dual_constraint_values.shape[1]
                if dual_values.shape != (ncons,):
                    raise ValueError(
                        "The dimensions of dual_values are not compatible with the other dual vectors",
                        dual_values.shape,
                    )

        self._xvalues = np.vstack([self._xvalues, x])
        self._objective_values = np.vstack([self._objective_values, fvalues])
        self._dual_objective_values = np.vstack(
            [self._dual_objective_values, dual_fvalues]
        )
        if self._dual_constraint_values is None:
            self._dual_constraint_values = dual_values
            if self._dual_constraint_values is not None:
                self._dual_constraint_values = np.reshape(
                    self._dual_constraint_values, (1, self._dual_constraint_values.size)
                )
        else:
            self._dual_constraint_values = np.vstack(
                [self._dual_constraint_values, dual_values]
            )

    def extreme_objective_vectors(self) -> np.ndarray:
        """Compute the set of extreme objective vectors.

        An extreme objective vector is a point in the objective space
        who reaches a minimum f value for one of the solutions.

        Returns
        -------
        np.ndarray
           A matrix of objective values of dimensions nobj x nobj:
           the row i corresponds to objective i.
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

        poly = Polyhedron(A=-self.halfspaces[:, 1:], b=self.halfspaces[:, 0])

        # Reload the set of hyperplanes to try removing redundant hyperplanes
        self._halfspaces = poly.halfspaces

        # Generate outer vertices
        outer_vertices = poly.V

        # In the case where no vertex is present, it means that the polyhedron does not possess any vertex or
        # has only (0, ..., 0) as a vertex, which will supposed to be the case for the moment
        if outer_vertices.size == 0:
            outer_vertices = np.zeros((1, self._dim))

        # Remove all vertices that are too close to each other
        # Taken from D. Dorfler original implementation of this algorithm
        indices = []
        for ind_i in range(len(outer_vertices)):
            for ind_j in range(ind_i + 1, len(outer_vertices)):
                if (
                    np.linalg.norm(outer_vertices[ind_j] - outer_vertices[ind_i])
                    <= MIN_DIST_OBJ_VECS
                ):
                    indices.append(ind_i)
                    break
        mask = np.ones(len(outer_vertices), dtype=bool)
        mask[indices] = False

        return outer_vertices[mask, ...]
