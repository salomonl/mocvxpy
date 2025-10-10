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

"""
Defines the constants used in the mocvxpy library
"""

# Below this distance, different objective vectors are
# considered as identical.
MIN_DIST_OBJ_VECS = 1e-10

# Below this tolerance, all hyperplane coordinates
# are rounded to zero.
MIN_TOL_HYPERPLANES = 1e-9

# The threshold used to compute the initial box for ADENA. The lower bound
# is the ideal objective vector of the problem and the upper bound an
# approximation of the nadir objective vector that is computed by taking
# the maximum along all coordinates of the nobj extreme points.
# The box is given by: [lb - ADENA_BOX_EXTENSION_TOL, ub + ADENA_BOX_EXTENSION_TOL]
ADENA_BOX_EXTENSION_TOL = 1e-4

# The maximum number of iterations allowed for ADENA
ADENA_MAX_ITER = 1000

# The maximum number of subproblems to solved allowed for ADENA
ADENA_MAX_PB_SOLVED = 10000

# The minimum stopping tolerance allowed for ADENA
ADENA_MIN_STOPPING_TOL = 1e-4

# The maximum number of iterations allowed for MONMO
MONMO_MAX_ITER = 10000

# The maximum number of subproblems to solved allowed for MONMO
MONMO_MAX_PB_SOLVED = 20000

# The minimum stopping tolerance allowed for MONMO
MONMO_MIN_STOPPING_TOL = 1e-6

# The maximum number of iterations allowed for MOVS
MOVS_MAX_ITER = 10000

# The minimum stopping tolerance allowed for MOVS
MOVS_MIN_STOPPING_TOL = 1e-6
