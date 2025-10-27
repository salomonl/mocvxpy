# MOCVXPY

## Introduction

MOCVXPY is a Python library for convex multiobjective and vector optimization, built on top of
CVXPY. It is distributed under the Apache 2.0 license.

Benefiting from single-objective counterpart library CVXPY features, it allows you to express your
problem in an intuitive and easy way that follows your mathematical formulation.

For example, here how you would solve a biobjective minimization problem whose set of optimal
solutions in the objective space correspond to a inferior quarter of a circle of radius 1 and center (1, 1).

``` python
import cvxpy as cp
import mocvxpy as mocp

# Solve:
# min f(x) = [x1, x2]
# s.t. (x1 - 1)**2 + (x2 - 1)**2 <= 1
#      x1, x2 >= 0

# Construct the problem
n = 2
x = mocp.Variable(n)
objectives = [cp.Minimize(x[0]), cp.Minimize(x[1])]
constraints = [x >= 0, cp.sum_squares(x - 1) <= 1]
pb = mocp.Problem(objectives, constraints)

# The optimal objective values are returned by calling the solve() method.
objective_values = pb.solve()
# You can collect the weights associated to each optimal solution,
# i.e., for convex optimization, each optimal solution is the optimal
# solution of a weighted sum problem min w' f(x) 
print(objectives[0].dual_values) # w1
print(objectives[1].dual_values) # w2
# The optimal values for x are stored in x.values
print(x.values)
```

As it is the case with most of available multiobjective optimization libraries, MOCVXPY will
generally only be able to compute a discrete representation of the optimal solution set, i.e., a
Pareto set approximation or a discrete set of efficient solutions. However, MOCVXPY's algorithms
guarantee the quality of the discrete solution set generated in relation to the entire optimal
solution set.

## Installation

MOCVXPY has the following dependencies:
- `Python` >= 3.12
- `cvxpy` >= 1.7.0
- `dask` >= 2025.7.0
- `distributed` >= 2025.7.0
- `pycddlib` >= 3.0.2
- `numpy` >= 2.3.0

Some of the examples require `Matplotlib`, one requires `Pandas`, and another requires the
commercial solver `Mosek`. The library used for the tests is `pytest`.

It is important to have a working installation of pycddlib. Its installation guide can be found at
the following [link](https://pycddlib.readthedocs.io/en/stable/). The easiest way to install the
library is to use `pip`. At the root of the project, type:

``` sh
python -m pip install -e .
```

MOCVXPY should be available on the `PyPy` platform soon.

## Testing

After installation, you can launch the test suite from outside the source directory.
You need `pytest` for that.

``` sh
pytest mocvxpy
```

## Contributing

MOCVXPY is a young project, there are still some bugs lurking. If you find one, please report it on
GitHub Issues.

If you have an efficient algorithm that you would like implemented in the library, you can open an
issue. However, this library is not open to:
- Algorithms without any mathematical convergence guarantees (in terms of sets).
- General multiobjective algorithms, not suited for (mixed-integer) convex optimization.

Particularly, this library will not provide evolutionary, particule-swarm, or bayesian optimization
algorithms. Indeed, there are other libraries that provide robust implementations of such
algorithms. If its application requires it, the practitioner is invited to consult for example
[pymoo](https://pymoo.org/), [pygmo](https://esa.github.io/pygmo2/), or
[BoTorch](https://botorch.org/). Conversely, MOCVXPY should be more efficient in terms of resolution
time and quality of the solution set for structured convex problems.

## Algorithms

MOCVXPY provides these three algorithms.
- [ADENA](https://doi.org/10.1016/j.ejor.2023.02.032)
- [MONMO](https://doi.org/10.1007/s10957-022-02045-8): see also
  [[1]](https://doi.org/10.1137%2F23M1574580) and [[2]](https://doi.org/10.1137%2F21M1458788)
- [MOVS](https://www.tandfonline.com/action/showCitFormats?doi=10.1080/10556788.2021.1880579)

MOVS is the default.

## Citation

If you find this work useful, please cite the following preprint
```
@misc{salomon2025mocvxpycvxpyextensionmultiobjective,
      title={MOCVXPY: a CVXPY extension for multiobjective optimization},
      author={Ludovic Salomon and Daniel Dörfler and Andreas Löhne},
      year={2025},
      eprint={2510.21010},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2510.21010},
}
```
and the article corresponding to the algorithm used in your application.
