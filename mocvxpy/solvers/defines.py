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
