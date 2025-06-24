import cdd
import numpy as np

from cdd import gmp
from fractions import Fraction

class OrderCone:
    """The order polyhedral cone of a multiobjective problem.

    An ordering polyhedral cone is defined by its H-representation:
    C = {y in R^m: A y >= 0}
    where:
    - m is the number of objectives of the problem.
    - A in R^{q x m}, such than rank(A) = m.

    Arguments
    ---------
    A: np.ndarray
       The matrix that defines the ordering cone
    """
    def __init__(self, A: np.ndarray) -> None:
        if len(A.shape) != 2:
            raise ValueError(
                f"A must be a two dimensional matrix. Get {A.shape}"
            )

        A_rank = np.linalg.matrix_rank(A)
        if A_rank != A.shape[1]:
            raise ValueError(
                f"A must be a full column-rank matrix. It has {A.shape[1]} columns but rank(A) = {A_rank}"
            )

        self._inequalities = A
        self._rays = None

    @property
    def inequalities(self) -> np.ndarray:
        """Returns: np.ndarray

        the inequalities that define the ordering cone.
        """
        return self._inequalities

    @property
    def rays(self) -> np.ndarray:
        """Returns: np.ndarray

        the rays that define the ordering cone.
        """
        if self._rays is not None:
            return self._rays

        # Compute its V-representation from its H-representation.
        try:
            halfspaces = [[0.0] + h.tolist() for h in self.inequalities]
            hrep = cdd.matrix_from_array(
                halfspaces, rep_type=cdd.RepType.INEQUALITY
            )
            cdd.matrix_canonicalize(hrep)
            vpoly = cdd.polyhedron_from_matrix(hrep)
            vrep = cdd.copy_generators(vpoly)
            self._rays = np.asarray([
                [vi for vi in ray[1:]]
                for ray in vrep.array
            ])
            return self._rays
        except RuntimeError:
            pass

        # Recompute its V-representation with exact precision
        halfspaces = [[0] + [Fraction(hi) for hi in h] for h in self.inequalities]
        hrep = gmp.matrix_from_array(
            halfspaces, rep_type=cdd.RepType.INEQUALITY
        )
        gmp.matrix_canonicalize(hrep)
        vpoly = gmp.polyhedron_from_matrix(hrep)
        vrep = gmp.copy_generators(vpoly)
        self._rays = np.asarray([
            [float(vi) for vi in ray[1:]]
            for ray in vrep.array
        ])
        return self._rays

def compute_order_cone_from_its_rays(D: np.ndarray) -> OrderCone:
    """Compute the H-representation of the order cone given its rays.

    Given C = cone(D), where each row of D represents a ray of the cone.
    Compute its H-representation, i.e., C under the form:
    C = {y: Z y >= 0}.

    Arguments
    ---------
    D: np.ndarray
       The matrix of rays of the order cone. Each row corresponds to one ray.

    Returns
    -------
    OrderCone:
       The ordering cone under its H-representation.
    """
    nobj = D.shape[1]
    if nobj <= 1:
        raise ValueError("The number of columns of D must be superior or equal to 2", nobj)
    nrays = D.shape[0]
    if nrays < nobj:
        raise ValueError(f"The number of rows of D ({nrays:d}) must be superior or equal to",
                         f"the number of columns of D ({nobj:d}).")

    v_representation = [[0.0] + ray.tolist() for ray in D]

    # Compute h-representation
    try:
        vmat = cdd.matrix_from_array(
            v_representation, rep_type=cdd.RepType.GENERATOR
        )
        cdd.matrix_canonicalize(vmat)
        vpoly = cdd.polyhedron_from_matrix(vmat)
        hmat = cdd.copy_inequalities(vpoly)
        cdd.matrix_redundancy_remove(hmat)
        Z = [z[1:] for z in hmat.array]
        C = OrderCone(np.asarray(Z))

        # Set directly its rays
        C._rays = np.copy(D)
        return C
    except RuntimeError:
        pass

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
        Z = [z[1:] for z in hmat]
    C = OrderCone(np.asarray(Z))

    # Set directly its rays
    C._rays = np.copy(D)
    return C
