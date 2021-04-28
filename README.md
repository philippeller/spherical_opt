# spherical_opt

Module providing spherically aware optimization routines

## Installation

`pip install --editable .`

## Usage

We minimize as an exmple a function `f` that takes as input a 4d array and outputs a scalar. Two of the 4 input dimesnions (indices 2 & 3) are angles on a sphere.

```python
from spherical_opt import spherical_opt

initial_points = np.zeros(4)

result = spherical_opt.spherical_opt(f,
                                     method="CRS2",
                                     initial_points=initial_points,
                                     spherical_indices=((2,3),))
```

Available methods are "Nelder-Mead" and "CRS2", for local and global optimization, respectively. More options, convergence criteria, and batch parallelization are available. This is the list of parameters the `spherical_opt` function accepts:

```
Parameters:
-----------
func : callable
    objective function
    if batch_size == 1, func should be a scalar function
    if batch_size >  1, func should be a vector function
method : string
    choices of 'Nelder-Mead' and 'CRS2'
inital_points : array
    providing the initial points for the algorithm, shape (N_points, N_dim)
spherical_indices : iterable of tuples
    indices of spherical coordinates in pairs of (azimuth, zenith) e.g.
    `[[0,1], [7,8]]` would identify indices 0 as azimuth and 1 as zenith as
    spherical coordinates and 7 and 8 another pair of independent spherical
    coordinates
param_boundaries : iterable of tuples
    hard param boundaries. For Cartesian coordinates only. Tuples should be of form
    `(param_index, (low_limit, high_limit))`. Use `None` for high or low limit
    to omit the check on that side. The optimizer will clip the param at param_index to
    the specified limits.
batch_size : int, optional 
    the number of new points proposed at each algorithm iteration
    batch_size > 1 is only supported for the CRS2 method
max_iter : int
    maximum number of iterations
max_calls : int
    maximum number of function calls
max_noimprovement : int
    break condition, maximum iterations without improvement
fstdthresh : float
    break condition, if std(f(p_i)) for all current points p_i drops below
    fstdthresh, minimization terminates
cstdthresh : array
    break condition, if std(p_i) for all non-spherical coordinates current
    points p_i drops below cstdthresh, minimization terminates, for
    negative values, coordinate will be ignored
sstdthresh : array
    break condition, if std(p_i) for all spherical coordinates current
    points p_i drops below sstdthresh, minimization terminates, for
    negative values, coordinate will be ignored
verbose : bool
rand : numpy random state or generator (optional)```
