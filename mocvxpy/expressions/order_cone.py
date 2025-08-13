import numpy as np

from mocvxpy.utilities.polyhedron import Polyhedron


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
            raise ValueError(f"A must be a two dimensional matrix. Get {A.shape}")

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
        ninequalities = self.inequalities.shape[0]
        poly = Polyhedron(A=-self.inequalities, b=np.zeros(ninequalities))
        self._rays = poly.generators[
            :, 1:
        ]  # No need to check if they are vertices or rays.

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
        raise ValueError(
            "The number of columns of D must be superior or equal to 2", nobj
        )
    nrays = D.shape[0]
    if nrays < nobj:
        raise ValueError(
            f"The number of rows of D ({nrays:d}) must be superior or equal to",
            f"the number of columns of D ({nobj:d}).",
        )

    # Compute its h-representation
    poly = Polyhedron(D=D)
    Z = poly.halfspaces[:, 1:]

    C = OrderCone(Z)

    # Set directly its rays
    C._rays = np.copy(D)
    return C


def polar_cone(C: OrderCone) -> OrderCone:
    """Compute the polar cone of an order cone, given by its H-representation.

    Arguments
    ---------
    C: OrderCone
       The ordering cone.

    Returns
    -------
    OrderCone:
       The polar cone.
    """
    # Compute the dual of C. Given C = {y: Z y >= 0},
    # the dual cone of C is given by: C+ = cone(Z^T)
    dual_C = compute_order_cone_from_its_rays(C.inequalities)

    # The polar cone is defined by: C* = - C+
    polar_C = OrderCone(-dual_C.inequalities)
    polar_C._rays = -dual_C.rays

    return polar_C
