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

from mocvxpy.solvers.parallel.monmopar_solver import MONMOParSolver
from mocvxpy.solvers.parallel.movspar_solver import MOVSParSolver
from mocvxpy.solvers.sequential.adena_solver import ADENASolver
from mocvxpy.solvers.sequential.monmo_solver import MONMOSolver
from mocvxpy.solvers.sequential.movs_solver import MOVSSolver

MO_PARALLEL_SOLVERS = ["MONMO", "MOVS"]

MO_PARALLEL_SOLVERS_MAP = {
    "MONMO": MONMOParSolver,
    "MOVS": MOVSParSolver,
}

MO_SEQUENTIAL_SOLVERS = ["ADENA", "MONMO", "MOVS"]

MO_SEQUENTIAL_SOLVERS_MAP = {
    "ADENA": ADENASolver,
    "MONMO": MONMOSolver,
    "MOVS": MOVSSolver,
}

VO_PARALLEL_SOLVERS = ["MONMO", "MOVS"]

VO_PARALLEL_SOLVERS_MAP = {
    "MONMO": MONMOParSolver,
    "MOVS": MOVSParSolver,
}

VO_SEQUENTIAL_SOLVERS = ["MONMO", "MOVS"]

VO_SEQUENTIAL_SOLVERS_MAP = {"MONMO": MONMOSolver, "MOVS": MOVSSolver}
