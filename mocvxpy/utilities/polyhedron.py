"""
NB: This class is inspired by bensolve-tools. It builds a simple and
robust interface around pycddlib to tackle the vertex
enumeration problem.
"""

import cdd
import numpy as np

from cdd import gmp
from fractions import Fraction
from mocvxpy.constants import MIN_TOL_HYPERPLANES
from typing import Optional, Set


class Polyhedron:
    """The polyhedron class.

    A polyhedron can be represented into two forms.
    - A H-representation:

    P = {x in R^n: A x <= b}

    where A in R^{m x n} and b in R^{m}
    - A V-representation:

    P = {x in R^n: x = V^T lambda + D^T mu: lambda >= 0, mu >= 0, sum lambda = 1}

    where each row of V represents a vertex of a polyhedron and
    each row of D represents a direction.

    Arguments:
    ---------
    A: np.ndarray, optional
        The inequality matrix.
    b: np.ndarray, optional.
        The rhs vector of the inequalities.
    eq_indexes: set{int}, optional.
        The set of linear equality indexes, i.e., {i in I: a_i^T x = b}.
    V: np.ndarray, optional.
        The matrix of vertices. Each row describes one vertex.
    D: np.ndarray, optional.
        The matrix of directions. Each row describes one direction.

    NB: The user cannot give both representations. Once entered,
    the polyhedron instance cannot be modified.
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        eq_indexes: Optional[Set[int]] = None,
        V: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ) -> None:
        if A is None and V is None and D is None:
            raise ValueError("One must provide H or V-representation parameters")
        if (A is not None or b is not None) and (V is not None or D is not None):
            raise ValueError(
                "One must provide exclusively H or V-representation parameters"
            )

        if A is None:
            # Check V-representation parameters
            if V is not None:
                ndim = V.ndim
                if ndim != 2:
                    raise ValueError("V must be a two-dimensional matrix")
                if V.shape[0] == 0 or V.shape[1] == 0:
                    raise ValueError(f"V must not be empty: dims = {V.shape}")
            if D is not None:
                ndim = D.ndim
                if ndim != 2:
                    raise ValueError("D must be a two-dimensional matrix")
                if D.shape[0] == 0 or D.shape[1] == 0:
                    raise ValueError(f"D must not be empty: dims = {D.shape}")
            if V is not None and D is not None:
                if V.shape[1] != D.shape[1]:
                    raise ValueError(
                        f"V {V.shape} and D {D.shape} have incompatible dimensions"
                    )
        else:
            # Check H-representation parameters
            if A.ndim != 2:
                raise ValueError("A must be a two-dimensional matrix")
            if A.shape[0] <= 0 or A.shape[1] <= 0:
                raise ValueError(f"A must not be empty: dims = {A.shape}")
            if b is None:
                raise ValueError("One must provide the rhs of the H-representation")
            if b.ndim != 1:
                raise ValueError("b is not a vector")
            if b.shape[0] != A.shape[0]:
                raise ValueError(
                    f"b {b.shape} and A {A.shape} have incompatible dimensions"
                )
            if np.any(b == np.inf):
                raise ValueError(
                    "One of the constraints of the H-representation does not have a finite bound"
                )
            if np.any(b == -np.inf):
                raise ValueError("The problem is infeasible")
            if eq_indexes is not None:
                for ind in eq_indexes:
                    if ind >= A.shape[0] or ind < 0:
                        raise ValueError(
                            f"The equality constraint index {ind} does not belong to the set of inequality constraints"
                        )

        self._halfspaces = None
        self._equality_indexes = None
        self._generators = None

        # Flag to precise if H-representation is computed using exact or floating point arithmetic
        # For the moment, the V-representation is always computed in exact arithmetic
        self._use_gmp_vpolyhedron = False
        self._vpolyhedron = None
        self._hpolyhedron = None

        if A is not None:
            self._halfspaces = np.column_stack((b, -A))
            self._equality_indexes = eq_indexes
            self._compute_vrepresentation()
        else:
            self._generators = []
            if V is not None:
                for vertex in V:
                    self._generators.append([1.0] + vertex.tolist())
            if D is not None:
                for ray in D:
                    self._generators.append([0.0] + ray.tolist())
            self._generators = np.asarray(self._generators)
            self._compute_hrepresentation()

    def _compute_hrepresentation(self) -> None:
        """Internal: must not be called by the user.

        Compute the H-representation of the polyhedron.
        """
        if self._generators is None:
            raise ValueError("The polyhedron does not have a valid V-representation")

        try:
            vmat = cdd.matrix_from_array(
                self._generators, rep_type=cdd.RepType.GENERATOR
            )
            cdd.matrix_canonicalize(vmat)
            self._vpolyhedron = cdd.polyhedron_from_matrix(vmat)
            hmat = cdd.copy_inequalities(self._vpolyhedron)
            cdd.matrix_redundancy_remove(hmat)
            self._equality_indexes = hmat.lin_set
            hmat = hmat.array
            vmat = vmat.array
        except RuntimeError:
            # Recompute but use the exact precision: note that it is slower
            v_exact_representation = []
            for vertex in self._generators:
                v_exact = [Fraction(vi) for vi in vertex]
                v_exact_representation.append(v_exact)
            vmat_exact = gmp.matrix_from_array(
                v_exact_representation, rep_type=cdd.RepType.GENERATOR
            )
            self._use_gmp_vpolyhedron = True
            gmp.matrix_canonicalize(vmat_exact)
            self._vpolyhedron = gmp.polyhedron_from_matrix(vmat_exact)
            hmat_exact = gmp.copy_inequalities(self._vpolyhedron)
            self._equality_indexes = hmat_exact.lin_set
            hmat = []
            for halfspace in hmat_exact.array:
                hmat.append([float(hi) for hi in halfspace])
            vmat = []
            for vertex in vmat_exact.array:
                vmat.append([float(vi) for vi in vertex])

        # Save the H-representation
        hmat = np.array(hmat)
        self._halfspaces = hmat

        # Reload the V-representation if redundant directions and/or
        # vertices have been eliminated
        vmat = np.array(vmat)
        self._generators = vmat

    def _compute_vrepresentation(self) -> None:
        """Internal: must not be called by the user.

        Compute the V-representation of the polyhedron.
        """
        if self._halfspaces is None:
            raise ValueError("The polyhedron does not have a valid H-representation")

        # Try removing redundant hyperplanes
        try:
            hrep = cdd.matrix_from_array(
                self._halfspaces, rep_type=cdd.RepType.INEQUALITY
            )
            if self._equality_indexes is not None:
                hrep.lin_set = self._equality_indexes
            cdd.matrix_canonicalize(hrep)
            self._halfspaces = []
            for halfspace in hrep.array:
                # Apply some rounding to try to mitigate numerical errors
                self._halfspaces.append(
                    [h if abs(h) >= MIN_TOL_HYPERPLANES else 0.0 for h in halfspace]
                )
            self._halfspaces = np.asarray(self._halfspaces)
            self._equality_indexes = hrep.lin_set
        except RuntimeError:
            pass

        # Generate vertices
        hrep_exact = []
        for halfspace in self._halfspaces:
            halfspace_exact = [Fraction(h) for h in halfspace]
            hrep_exact.append(halfspace_exact)
        hrep_exact = gmp.matrix_from_array(hrep_exact, rep_type=cdd.RepType.INEQUALITY)
        if self._equality_indexes is not None:
            hrep_exact.lin_set = self._equality_indexes
        self._hpolyhedron = gmp.polyhedron_from_matrix(hrep_exact)
        vrep = gmp.copy_generators(self._hpolyhedron)

        # Convert into inexact floating point.
        self._generators = [[float(vi) for vi in vertex] for vertex in vrep.array]
        self._generators = np.array(self._generators)

    @property
    def halfspaces(self) -> np.ndarray:
        """
        Returns
        -------
        A set of halfspaces {x in R^n: Ax <= b} under the form [b -A].
        """
        if self._halfspaces is None:
            self._compute_hrepresentation()

        return self._halfspaces

    @property
    def equality_indexes(self) -> Set[int]:
        """
        Returns
        -------
        A set of equality indexes of the H-representation, i.e.,
        {i in I: a_i^T x = b}.
        """
        if self._equality_indexes is None:
            self._compute_hrepresentation()

        return self._equality_indexes

    @property
    def generators(self) -> np.ndarray:
        """
        Returns
        -------
        A set of generators of the V-representation.
        Each row of the matrix starts by a 1 (for a vertex) or 0 (for
        a ray) followed by the coordinates of the corresponding vertex
        or ray.
        """
        if self._generators is None:
            self._compute_vrepresentation()

        return self._generators

    @property
    def A(self) -> np.ndarray:
        """
        Returns
        -------
        The matrix A of the H-representation.
        """
        if self._halfspaces is None:
            self._compute_hrepresentation()

        return -self._halfspaces[:, 1:]

    @property
    def b(self) -> np.ndarray:
        """
        Returns
        -------
        The rhs vector b of the H-representation.
        """
        if self._halfspaces is None:
            self._compute_hrepresentation()

        return -self._halfspaces[:, 0]

    @property
    def V(self) -> np.ndarray:
        """
        Returns
        -------
        The matrix V of the V-representation.
        """
        if self._generators is None:
            self._compute_vrepresentation()

        return self._generators[self._generators[:, 0] == 1, 1:]

    @property
    def D(self) -> np.ndarray:
        """
        Returns
        -------
        The matrix of directions/rays of the V-representation.
        """

        if self._generators is None:
            self._compute_vrepresentation()

        return self._generators[self._generators[:, 0] == 0, 1:]

    @property
    def adjacent_vertex_list(self):
        """
        Returns
        -------
        The list of adjacent vertices. Each element of the list
        is a list of indexes of adjacent vertices.

        Must be used in combination with self.generators()
        """
        if self._hpolyhedron is not None:
            return gmp.copy_adjacency(self._hpolyhedron)

        if self._vpolyhedron is not None:
            if self._use_gmp_vpolyhedron:
                return gmp.copy_input_adjacency(self._vpolyhedron)
            return cdd.copy_input_adjacency(self._vpolyhedron)

        raise RuntimeError()

    @property
    def incident_vertex_list(self):
        """
        Returns
        -------
        The list of incident faces to vertices. Each element of the
        list is a list of indexes of incident faces.

        Must be used in combination with self.generators() and self.halfspaces()
        """
        if self._hpolyhedron is not None:
            return gmp.copy_incidence(self._hpolyhedron)

        if self._vpolyhedron is not None:
            if self._use_gmp_vpolyhedron:
                return gmp.copy_input_incidence(self._vpolyhedron)
            return cdd.copy_input_incidence(self._vpolyhedron)

        raise RuntimeError()

    @property
    def adjacent_faces_list(self):
        """
        Returns
        -------
        The list of adjacent faces to faces. Each element of the
        list is a list of indexes of adjacent faces.

        Must be used in combination with self.halfspaces()
        """
        if self._hpolyhedron is not None:
            return gmp.copy_input_adjacency(self._hpolyhedron)

        if self._vpolyhedron is not None:
            if self._use_gmp_vpolyhedron:
                return gmp.copy_adjacency(self._vpolyhedron)
            return cdd.copy_adjacency(self._vpolyhedron)

        raise RuntimeError()

    @property
    def incident_faces_list(self):
        """
        Returns
        -------
        The list of incident vectors to faces. Each element of the
        list is a list of indexes of incident vectors.

        Must be used in combination with self.halfspaces() and self.generators()
        """
        if self._hpolyhedron is not None:
            return gmp.copy_input_incidence(self._hpolyhedron)

        if self._vpolyhedron is not None:
            if self._use_gmp_vpolyhedron:
                return gmp.copy_incidence(self._vpolyhedron)
            return cdd.copy_incidence(self._vpolyhedron)

        raise RuntimeError()
