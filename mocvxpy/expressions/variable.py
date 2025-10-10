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
    lambda self, vals: setattr(self, "_values", vals),
    doc="""The numeric values of the variable.

         Each value corresponds to an optimal value of a multiobjective problem.
         """,
)
