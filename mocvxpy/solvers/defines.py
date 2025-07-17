from mocvxpy.solvers.sequential.adena_solver import ADENASolver
from mocvxpy.solvers.sequential.monmo_solver import MONMOSolver
from mocvxpy.solvers.sequential.movs_solver import MOVSSolver

MO_SEQUENTIAL_SOLVERS = ["ADENA", "MONMO", "MOVS"]

MO_SEQUENTIAL_SOLVERS_MAP = {
    "ADENA": ADENASolver,
    "MONMO": MONMOSolver,
    "MOVS": MOVSSolver,
}

VO_SEQUENTIAL_SOLVERS = ["MONMO", "MOVS"]

VO_SEQUENTIAL_SOLVERS_MAP = {"MONMO": MONMOSolver, "MOVS": MOVSSolver}
