import cvxpy as cp

"""The optimization variables in a problem.

It is an alias of the cvxpy.Variable class with new properties for the
multiobjective case.

NB: For special variable types (e.g., complex hermitian matrix variables),
cvxpy cannot handle inherited subclass of cp.Variable when
making transformations to the expression tree of the problem.
For these reasons, we resort to this ad-hoc monkey-patching.
"""
Variable = cp.Variable
Variable._values = None
Variable.values = property(
    lambda self: self._values,
    lambda self, vals: setattr(self, '_values', vals),
    doc="""The numeric values of the variable.

         Each value corresponds to an optimal value of a multiobjective problem.
         """,
)
