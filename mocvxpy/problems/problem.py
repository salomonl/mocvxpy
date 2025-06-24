"""
NB: This class is inspired by a large part of the code used in the
class Problem from the cvxpy package.
"""

import cvxpy as cp
import numpy as np

from cvxpy.constraints import Equality
from cvxpy.utilities.deterministic import unique_list
from cvxpy.utilities import performance_utils as perf
from typing import Dict, List, Optional, Union


class Problem:
    """A convex multiobjective optimization problem.

    Arguments
    ---------
    objectives : list[Minimize or Maximize]
        The problem's objectives.
    constraints : list
        The constraints on the problem variables.
    """

    def __init__(
        self,
        objectives: List[Union[cp.Minimize, cp.Maximize]],
        constraints: Optional[List[cp.Constraint]] = None,
    ) -> None:
        if constraints is None:
            constraints = []

        # Check that objective is Minimize or Maximize.
        for i, objective in enumerate(objectives):
            if not isinstance(objective, (cp.Minimize, cp.Maximize)):
                raise cp.error.DCPError(
                    f"Objective{i} is not in cvxpy.Minimize or cvxpy.Maximize."
                )

        # Constraints and objective are immutable.
        # We add a values field to them by monkey patching
        self._objectives = []
        for objective in objectives:
            obj = objective
            obj._values = None
            obj.values = property(lambda self: self._values)
            self._objectives.append(obj)

        self._constraints = []
        for constraint in constraints:
            cstr = constraint
            cstr._values = None
            cstr._values = property(lambda self: self._values)
            self._constraints.append(cstr)

        self._status: Optional[str] = None
        self._objective_values = None
        self._solutions = None

    @property
    def objective_values(self) -> Optional[np.ndarray]:
        """np.ndarray: The optimal values of the objectives from the last
            time the problem was solved.
        (or None if not solved).
        """
        if self._objectives is None:
            return None
        else:
            return self._objective_values

    @property
    def status(self) -> str:
        """str : The status from the last time the problem was solved; one
        of optimal, infeasible, or unbounded (with or without
        suffix inaccurate).
        """
        return self._status

    @property
    def solutions(self):
        """Solution : The solutions from the last time the problem was solved."""
        return self._solutions

    @property
    def objectives(self) -> List[Union[cp.Minimize, cp.Maximize]]:
        """Minimize or Maximize : The problem's objectives.

        Note that the objective cannot be reassigned after creation,
        and modifying the objective after creation will result in
        undefined behavior.
        """
        return self._objectives

    @property
    def constraints(self) -> List[cp.Constraint]:
        """A shallow copy of the problem's constraints.

        Note that constraints cannot be reassigned, appended to, or otherwise
        modified after creation, except through parameters.
        """
        return self._constraints[:]

    @property
    def param_dict(self):
        """
        Expose all parameters as a dictionary
        """
        return {parameters.name(): parameters for parameters in self.parameters()}

    @property
    def var_dict(self) -> Dict[str, cp.Variable]:
        """
        Expose all variables as a dictionary
        """
        return {variable.name(): variable for variable in self.variables()}

    @perf.compute_once
    def is_dcp(self, dpp: bool = False) -> bool:
        """Does the problem satisfy DCP rules?

        Arguments
        ---------
        dpp : bool, optional
            If True, enforce the disciplined parametrized programming (DPP)
            ruleset; only relevant when the problem involves Parameters.
            DPP is a mild restriction of DCP. When a problem involving
            Parameters is DPP, subsequent solves can be much faster than
            the first one. For more information, consult the documentation at

            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Returns
        -------
        bool
            True if the Expression is DCP, False otherwise.
        """
        return all(expr.is_dcp(dpp) for expr in self.constraints + self.objectives)

    @perf.compute_once
    def _max_ndim(self) -> int:
        """
        Returns the maximum number of dimensions of any argument in the problem.
        """
        return max(expr._max_ndim() for expr in self.constraints + self.objectives)

    @perf.compute_once
    def _supports_cpp(self) -> bool:
        """
        Returns True if all the arguments in the problem support cpp backend.
        """
        return all(
            expr._all_support_cpp() for expr in self.constraints + self.objectives
        )

    @perf.compute_once
    def is_dgp(self, dpp: bool = False) -> bool:
        """Does the problem satisfy DGP rules?

        Arguments
        ---------
        dpp : bool, optional
            If True, enforce the disciplined parametrized programming (DPP)
            ruleset; only relevant when the problem involves Parameters.
            DPP is a mild restriction of DGP. When a problem involving
            Parameters is DPP, subsequent solves can be much faster than
            the first one. For more information, consult the documentation at

            https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Returns
        -------
        bool
            True if the Expression is DGP, False otherwise.
        """
        return all(expr.is_dgp(dpp) for expr in self.constraints + self.objectives)

    @perf.compute_once
    def is_dqcp(self) -> bool:
        """Does the problem satisfy the DQCP rules?"""
        return all(expr.is_dqcp() for expr in self.constraints + self.objectives)

    @perf.compute_once
    def is_dpp(self, context: str = "dcp") -> bool:
        """Does the problem satisfy DPP rules?

        DPP is a mild restriction of DGP. When a problem involving
        Parameters is DPP, subsequent solves can be much faster than
        the first one. For more information, consult the documentation at

        https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

        Arguments
        ---------
        context : str
            Whether to check DPP-compliance for DCP or DGP; ``context`` should
            be either ``'dcp'`` or ``'dgp'``. Calling ``problem.is_dpp('dcp')``
            is equivalent to ``problem.is_dcp(dpp=True)``, and
            `problem.is_dpp('dgp')`` is equivalent to
            `problem.is_dgp(dpp=True)`.

        Returns
        -------
        bool
            Whether the problem satisfies the DPP rules.
        """
        if context.lower() == "dcp":
            return self.is_dcp(dpp=True)
        elif context.lower() == "dgp":
            return self.is_dgp(dpp=True)
        else:
            raise ValueError("Unsupported context ", context)

    @perf.compute_once
    def is_qp(self) -> bool:
        """Is problem a quadratic program?"""
        for c in self.constraints:
            if not (isinstance(c, (Equality, cp.Zero)) or c.args[0].is_pwl()):
                return False
        for var in self.variables():
            if var.is_psd() or var.is_nsd():
                return False
        if not self.is_dcp():
            return False

        for objective in self.objectives:
            if not objective.args[0].is_qpwa():
                return False

        return True

    @perf.compute_once
    def is_mixed_integer(self) -> bool:
        return any(
            v.attributes["boolean"] or v.attributes["integer"] for v in self.variables()
        )

    @perf.compute_once
    def variables(self) -> List[cp.Variable]:
        """Accessor method for variables.

        Returns
        -------
        list of :class:`~cvxpy.expressions.variable.Variable`
            A list of the variables in the problem.
        """
        vars_ = []
        for objective in self.objectives:
            vars_ += objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        return unique_list(vars_)

    @perf.compute_once
    def parameters(self):
        """Accessor method for parameters.

        Returns
        -------
        list of :class:`~cvxpy.expressions.constants.parameter.Parameter`
            A list of the parameters in the problem.
        """
        params = []
        for objective in self.objectives:
            params += objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        return unique_list(params)

    @perf.compute_once
    def constants(self) -> List[cp.Constant]:
        """Accessor method for constants.

        Returns
        -------
        list of :class:`~cvxpy.expressions.constants.constant.Constant`
            A list of the constants in the problem.
        """
        const_dict = {}
        constants_ = []
        for objective in self.objectives:
            constants_ = objective.constants()
        for constr in self.constraints:
            constants_ += constr.constants()
        # Note that numpy matrices are not hashable, so we use the built-in
        # function "id"
        const_dict = {id(constant): constant for constant in constants_}
        return list(const_dict.values())

    def solve(self, *args, **kwargs):
        pass

    def _clear_solution(self) -> None:
        for v in self.variables():
            v.save_value(None)
        for c in self.constraints:
            for dv in c.dual_variables:
                dv.save_value(None)
        self._objective_values = None
        self._status = None
        self._solutions = None

    def __str__(self) -> str:
        lines = []
        for objective in self.objectives:
            lines.append(str(objective))

        if len(self.constraints) == 0:
            return "\n".join(lines)

        subject_to = "subject to "
        lines += subject_to + str(self.constraints[0])
        for constr in self.constraints[1:]:
            lines += [len(subject_to) * " " + str(constr)]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "Problem(%s, %s)" % (repr(self.objectives), repr(self.constraints))

    @staticmethod
    def is_constant() -> bool:
        return False