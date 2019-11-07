# -*- coding: utf-8 -*-

'''
Module for Optimization of functions with spherical parameters
'''

from __future__ import absolute_import, division, print_function

__author__ = 'P. Eller'
__license__ = '''Copyright 2019 Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.'''

import numpy as np
import copy

SPHER_T = np.dtype([
    ('zen', np.float32),
    ('az', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('sinzen', np.float32),
    ('coszen', np.float32),
    ('sinaz', np.float32),
    ('cosaz', np.float32),
])
"""type to store spherical coordinates and handy quantities"""

def fill_from_spher(s):
    """Fill in the remaining values in SPHER_T type giving the two angles `zen` and
    `az`.
    Parameters
    ----------
    s : SPHER_T
    """
    s['sinzen'] = np.sin(s['zen'])
    s['coszen'] = np.cos(s['zen'])
    s['sinaz'] = np.sin(s['az'])
    s['cosaz'] = np.cos(s['az'])
    s['x'] = s['sinzen'] * s['cosaz']
    s['y'] = s['sinzen'] * s['sinaz']
    s['z'] = s['coszen']


def fill_from_cart(s_vector):
    """Fill in the remaining values in SPHER_T type giving the cart, coords. `x`, `y`
    and `z`.
    Parameters
    ----------
    s_vector : SPHER_T
    """
    for s in s_vector:
        radius = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
        if radius > 0.:
            # make sure they're length 1
            s['x'] /= radius
            s['y'] /= radius
            s['z'] /= radius
            s['az'] = np.arctan2(s['y'], s['x']) % (2 * np.pi)
            s['coszen'] = s['z']
            s['zen'] = np.arccos(s['coszen'])
            s['sinzen'] = np.sin(s['zen'])
            s['sinaz'] = np.sin(s['az'])
            s['cosaz'] = np.cos(s['az'])
        else:
            #print 'zero length'
            s['z'] = 1.
            s['az'] = 0.
            s['zen'] = 0.
            s['coszen'] = 1.
            s['sinzen'] = 0.
            s['cosaz'] = 1.
            s['sinaz'] = 0.


def reflect(old, centroid, new):
    """Reflect the old point around the centroid into the new point on the sphere.
    Parameters
    ----------
    old : SPHER_T
    centroid : SPHER_T
    new : SPHER_T
    """
    x = old['x']
    y = old['y']
    z = old['z']

    ca = centroid['cosaz']
    sa = centroid['sinaz']
    cz = centroid['coszen']
    sz = centroid['sinzen']

    new['x'] = (
        2*ca*cz*sz*z
        + x*(ca*(-ca*cz**2 + ca*sz**2) - sa**2)
        + y*(ca*sa + sa*(-ca*cz**2 + ca*sz**2))
    )
    new['y'] = (
        2*cz*sa*sz*z
        + x*(ca*sa + ca*(-cz**2*sa + sa*sz**2))
        + y*(-ca**2 + sa*(-cz**2*sa + sa*sz**2))
    )
    new['z'] = 2*ca*cz*sz*x + 2*cz*sa*sz*y + z*(cz**2 - sz**2)

    fill_from_cart(new)

def centroid(cart_coords, sph_coord):
    '''
    Compute centroid of two or more points
    '''
    centroid_sph = np.zeros_like(sph_coord[0])
    for dim in ['x', 'y', 'z']:
        centroid_sph[dim] = np.sum(sph_coord[dim])/sph_coord.shape[0]
    fill_from_cart(centroid_sph)
    centroid_cart = np.sum(cart_coords, axis=0)/cart_coords.shape[0]
    
    return centroid_cart, centroid_sph

def angular_dist(p1, p2): # theta1, theta2, phi1, phi2):
    '''
    calculate the angular distance between two directions in spherical coords
    '''
    return np.arccos(p1['coszen'] * p2['coszen'] + p1['sinzen'] * p2['sinzen'] * np.cos(p1['az'] - p2['az']))


def spherical_opt(func, method, initial_points, spherical_indices=[], max_iter=10000, max_calls=None, max_noimprovement=1000, fstd=1e-1, cstd=None, sstd=None, verbose=False, meta=False, rand=None):
    '''spherical minimization
    Parameters:
    -----------
    func : callable
        objective function
    method : string
        choices of 'Nelder-Mead' and 'CRS2'
    inital_points : array
        providing the initial points for the algorithm, shape (N_points, N_dim)
    spherical_indices : iterable of tuples
        indices of spherical coordinates in pairs of (azmiuth, zenith)
        e.g. `[[0,1], [7,8]]` would identify indices 0 as azimuth and 1 as zenith as spherical coordinates
        and 7 and 8 another pair of independent spherical coordinates
    max_iter : int
        maximum number of iterations
    max_calls : int
        maximum number of function calls
    max_noimprovement : int
        break condition, maximum iterations without improvement
    fstd : float
        break condition, if std(f(p_i)) for all current points p_i droppes below fstd, minimization terminates
    cstd : array
        break condition, if std(p_i) for all non-spherical coordinates current points p_i droppes below cstd, minimization terminates,
        for negative values, coordinate will be ignored
    fstd : array
        break condition, if std(p_i) for all spherical coordinates current points p_i droppes below sstd, minimization terminates,
        for negative values, coordinate will be ignored
    verbose : bool
    rand : numpy random state (optional)

    Notes
    -----
    CRS2 [1] is a variant of controlled random search (CRS, a global
    optimizer) with faster convergence than CRS.

    Refrences
    ---------
    .. [1] P. Kaelo, M.M. Ali, "Some variants of the controlled random
       search algorithm for global optimization," J. Optim. Theory Appl.,
       130 (2) (2006), pp. 253-264.
    '''
    if not method in ['Nelder-Mead', 'CRS2']:
        raise ValueError('Unknown method %s, choices are Nelder-Mead or CRS2'%method)

    if rand is None:
        rand = np.random.RandomState()
    
    #REPORT_AFTER = 100
    
    n_points, n_dim = initial_points.shape
    n_spher = len(spherical_indices)
    n_cart = n_dim - 2 * n_spher

    sdevs = - np.ones(n_spher)
    cdevs = - np.ones(n_cart)

    if cstd is not None:
        assert len(cstd) == n_cart, 'Std-dev stopping values for cartesian coordinates must have length equal to number of cartesian coordinates'
        cstd = np.array(cstd)

    if sstd is not None:
        assert len(sstd) == n_spher, 'Std-dev stopping values for spherical coordinates must have length equal to number of spherical coordinate pairs'
        sstd = np.array(sstd)

    if method == 'Nelder-Mead':
        assert n_points == n_dim + 1, 'Nelder-Mead will need n+1 points for an n-dimensional function'

    if method == 'CRS2':
        assert n_points > n_dim, 'CRS will need more points than dimesnsions'
        if n_points < 10 * n_dim:
            print('WARNING: number of points is very low')

        if meta:
            meta_dict = {}
            meta_dict['num_simplex_successes'] = 0      
            meta_dict['num_mutation_successes'] = 0     
            meta_dict['num_failures'] = 0       
    if meta:
        if cstd is not None:
            meta_dict['cstd_met_at_iter'] = np.full(len(cstd), -1)
        if sstd is not None:
            meta_dict['sstd_met_at_iter'] = np.full(len(sstd), -1)

    
    all_spherical_indices = [idx for sp in spherical_indices for idx in sp]
    all_azimuth_indices = [sp[0] for sp in spherical_indices]
    all_zenith_indices = [sp[1] for sp in spherical_indices]
    all_cartesian_indices = list(set(range(n_dim)) ^ set(all_spherical_indices))
    
    # first thing, pack the points into separate cartesian and spherical coordinates
    fvals = np.empty(shape=(n_points,))
    for i in range(n_points):
        fvals[i] = func(initial_points[i])
    
    s_cart = initial_points[:, all_cartesian_indices]
    #print(s_cart)
    s_spher = np.zeros(shape=(n_points, n_spher), dtype=SPHER_T)
    s_spher['az'] = initial_points[:, all_azimuth_indices]
    s_spher['zen'] = initial_points[:, all_zenith_indices]
    fill_from_spher(s_spher)
    
    # the array containing points in the original form
    x = copy.copy(initial_points)
    
    def create_x(x_cart, x_spher):
        '''Patch Cartesian and spherical coordinates back together into one array for function calls'''
        x = np.empty(shape=n_dim)
        x[all_cartesian_indices] = x_cart
        x[all_azimuth_indices] = x_spher['az']
        x[all_zenith_indices] = x_spher['zen']
        return x
    
    best_fval = np.min(fvals)
    best_idx = 0
    no_improvement_counter = -1
    n_calls = n_points
    stopping_flag = -1

    # minimizer loop
    for iter_num in range(max_iter+1):
        #print(iter_num)

        if max_calls and n_calls >= max_calls:
            stopping_flag = 0
            break                

        # break condition 2
        if max_noimprovement and no_improvement_counter > max_noimprovement:
            stopping_flag = 2
            break

        # break condition 1
        if np.std(fvals) < fstd:
            stopping_flag = 1
            break

        # break condition 3
        if cstd is not None or sstd is not None:
            # ToDo: stddev in spherical coords.
            if cstd is not None:
                cdevs = np.std(s_cart, axis=0)
                converged = cdevs[cstd>0] < cstd[cstd>0]
                if meta:
                    mask = np.logical_and(meta_dict['cstd_met_at_iter'] < 0, converged)
                    meta_dict['cstd_met_at_iter'][mask] = iter_num
                converged = np.all(converged)
            else:
                converged = True

            if sstd is not None:
                for i, std in enumerate(sstd):
                    if std > 0:
                        _, cent = centroid(np.empty([0,0]), s_spher)
                        deltas = angular_dist(s_spher, cent)
                        dev = np.sqrt(np.sum(np.square(deltas))/(n_points - 1))
                        sdevs[i] = dev
                        converged = converged and dev < std
                        if meta:
                            if meta_dict['sstd_met_at_iter'] < 0 and dev < std:
                                meta_dict['sstd_met_at_iter'] = iter_num
                    else:
                        sdevs[i] = -1
            if converged:
                stopping_flag = 3
                break
           
        sorted_idx = np.argsort(fvals)
        worst_idx = sorted_idx[-1]
        best_idx = sorted_idx[0]

        new_best_fval = fvals[best_idx]
        if new_best_fval < best_fval:
            best_fval = new_best_fval
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
    
        if method == 'CRS2':

            # choose n_dim random points but not best
            choice = rand.choice(n_points - 1, n_dim, replace=False)
            choice[choice >= best_idx] += 1
            
            # --- STEP 1: Reflection ---

            # centroid of choice except N+1, but including best
            centroid_indices = copy.copy(choice)
            centroid_indices[-1] = best_idx
            centroid_cart, centroid_spher = centroid(s_cart[centroid_indices], s_spher[centroid_indices])
 
            # reflect point
            reflected_p_cart = 2 * centroid_cart - s_cart[choice[-1]]
            reflected_p_spher = np.zeros(n_spher, dtype=SPHER_T)
            reflect(s_spher[choice[-1]], centroid_spher, reflected_p_spher)
            reflected_p = create_x(reflected_p_cart, reflected_p_spher)

            new_fval = func(reflected_p)
            n_calls += 1

            if new_fval < fvals[worst_idx]:
                # found better point
                s_cart[worst_idx] = reflected_p_cart
                s_spher[worst_idx] = reflected_p_spher
                x[worst_idx] = reflected_p
                fvals[worst_idx] = new_fval
                meta_dict['num_simplex_successes'] += 1
                continue
 
            # --- STEP 2: Mutation ---
                
            w = rand.uniform(0, 1, n_cart)
            mutated_p_cart = (1 + w) * s_cart[best_idx] - w * reflected_p_cart

            # first reflect at best point
            help_p_spher = np.zeros(n_spher, dtype=SPHER_T)
            reflect(reflected_p_spher, s_spher[best_idx], help_p_spher)
            mutated_p_spher = np.zeros_like(help_p_spher)
            # now do a combination of best and reflected point with weight w
            for dim in ['x', 'y', 'z']:
                w = rand.uniform(0, 1, n_spher)
                mutated_p_spher[dim] = (1 - w) * s_spher[best_idx][dim] + w * help_p_spher[dim]
            fill_from_cart(mutated_p_spher)

            mutated_p = create_x(mutated_p_cart, mutated_p_spher)
            
            new_fval = func(mutated_p)
            n_calls += 1

            if new_fval < fvals[worst_idx]:
                # found better point
                s_cart[worst_idx] = mutated_p_cart
                s_spher[worst_idx] = mutated_p_spher
                x[worst_idx] = mutated_p
                fvals[worst_idx] = new_fval
                meta_dict['num_mutation_successes'] += 1
                continue

            # if we get here no method was successful in replacing worst point -> start over
            meta_dict['num_failures'] += 1      

            
        elif method == 'Nelder-Mead':
            
            # --- STEP 1: Reflection ---
            if verbose: print('reflect')
            # centroid of choice except N+1, but including best
            centroid_indices = sorted_idx[:-1]
            centroid_cart, centroid_spher = centroid(s_cart[centroid_indices], s_spher[centroid_indices])
 
            # reflect point
            reflected_p_cart = 2 * centroid_cart - s_cart[worst_idx]
            reflected_p_spher = np.zeros(n_spher, dtype=SPHER_T)
            reflect(s_spher[worst_idx], centroid_spher, reflected_p_spher)
            reflected_p = create_x(reflected_p_cart, reflected_p_spher)
            reflected_fval = func(reflected_p)
            n_calls += 1

            if reflected_fval < fvals[sorted_idx[-2]] and reflected_fval >= fvals[best_idx]:
                # found better point
                s_cart[worst_idx] = reflected_p_cart
                s_spher[worst_idx] = reflected_p_spher
                x[worst_idx] = reflected_p
                fvals[worst_idx] = reflected_fval
                continue
                
            # --- STEP 2: Expand ---
                
            if reflected_fval < fvals[best_idx]:
                if verbose: print('expand')

                # essentially reflect again
                expanded_p_spher = np.zeros(n_spher, dtype=SPHER_T)
                reflect(centroid_spher, reflected_p_spher, expanded_p_spher)
                expanded_p_cart =  2. * reflected_p_cart - centroid_cart
                expanded_p = create_x(expanded_p_cart, expanded_p_spher)
                expanded_fval = func(expanded_p)
                n_calls += 1
                
                if expanded_fval < reflected_fval:
                    s_cart[worst_idx] = expanded_p_cart
                    s_spher[worst_idx] = expanded_p_spher
                    x[worst_idx] = expanded_p
                    fvals[worst_idx] = expanded_fval
                else:
                    s_cart[worst_idx] = reflected_p_cart
                    s_spher[worst_idx] = reflected_p_spher
                    x[worst_idx] = reflected_p
                    fvals[worst_idx] = reflected_fval
                continue

            # --- STEP 3: Contract ---
                
            if reflected_fval < fvals[worst_idx]:
                if verbose: print('contract (outside)')
                contracted_p_cart, contracted_p_spher = centroid(np.vstack([centroid_cart, reflected_p_cart]), np.vstack([centroid_spher, reflected_p_spher]))
                contracted_p = create_x(contracted_p_cart, contracted_p_spher)
                contracted_fval = func(contracted_p)
                n_calls += 1
                if contracted_fval < reflected_fval:
                    s_cart[worst_idx] = contracted_p_cart
                    s_spher[worst_idx] = contracted_p_spher
                    x[worst_idx] = contracted_p
                    fvals[worst_idx] = contracted_fval
                    continue
            else:
                if verbose: print('contract (inside)')
                contracted_p_cart, contracted_p_spher = centroid(np.vstack([centroid_cart, s_cart[worst_idx]]), np.vstack([centroid_spher, s_spher[worst_idx]]))
                contracted_p = create_x(contracted_p_cart, contracted_p_spher)
                contracted_fval = func(contracted_p)
                n_calls += 1
                if contracted_fval < fvals[worst_idx]:
                    s_cart[worst_idx] = contracted_p_cart
                    s_spher[worst_idx] = contracted_p_spher
                    x[worst_idx] = contracted_p
                    fvals[worst_idx] = contracted_fval
                    continue

            # --- STEP 4: Shrink ---
            if verbose: print('shrink')
                
            for idx in range(n_points):
                if not idx == best_idx:
                    s_cart[idx], s_spher[idx] = centroid(s_cart[[best_idx, idx]], s_spher[[best_idx, idx]])
                    x[idx] = create_x(s_cart[idx], s_spher[idx])
                    fvals[idx] = func(x[idx])
                    n_calls += 1


    if meta:
        meta_dict['no_improvement_counter'] = no_improvement_counter
        meta_dict['fstd'] = np.std(fvals)
        if cstd is not None:
            meta_dict['cstd'] = cdevs
        if sstd is not None:
            meta_dict['sstd'] = np.array(sdevs)
        

    opt_meta = {}
    opt_meta['stopping_flag'] = stopping_flag
    opt_meta['n_calls'] = n_calls
    opt_meta['nit'] = iter_num
    opt_meta['method'] = method
    opt_meta['fun'] = fvals[best_idx]
    opt_meta['x'] = x[best_idx]
    opt_meta['final_simplex'] = [x, fvals]
    opt_meta['success'] = stopping_flag > 0
    if meta:
        opt_meta['meta'] = meta_dict

    return opt_meta


