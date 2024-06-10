import numpy as np
from scipy.integrate import quad
from tqdm import tqdm
from numba import jit

import sys
sys.path.append("../cython/")
import lib_strawberry as stl

class ParticleAssigner:
    '''
    ParticleAssigner class. This class is designed to assign particles to a structure given a potential surface and local acceleration defining the boosted potential (for details see: https://arxiv.org/abs/2107.13008). 
    
    The underlying algorithm relies on traveling an ensemble of connected particles in search of a saddle point in the boosted potential. It is designed to be agnostic of the particular geometry of the problem ans simply requires knowledge of the connectivity between nodes/particles. In general this means that the algorithm requires a list of the nearest neighbours to each particle to segement the different groups. 
    
    The main method of this class is 'segement' which selects all the connected particles conected to particle 'i0' given a local acceleration 'acc0'.
    
    Parameters:
    ----------
    ngbs: (list of arrays of int) list of neighbours to each particle, these take the form of arrays of indices pointing to the corresponding neighbours
    pot: (array of floats) global gravitation potential evaluated at the position of each particle
    pos: (array of d-floats) positions of of particles in d-dimensional space.
    Lbox: (float) simulation boxsize
    Omega_Lambda: (float) Density of dark energy in units of the critical density (default to Einstein-de Sitter Omega_Lambda = 0.)
    scale_factor: (float) Snapshot scale factor 'a' (defaults to a = 1.)
    substruct: (bool) switch enabling the output of substructure properties
    no_binding: (bool) switch turning off the binding check
    verbose: (bool) switch toggling verbosity
    
    Methods:
    ----------
    - set_x0: sets the reference position to calculate the boosted potential
    
    - set_acc0: sets the reference acceleration to calculate the boosted potential
    
    - phi_boost: returns the boosted potential at the postion of particle i
    
    - subfind_neighbours: modifies the neighbour's list to correspond to a definition equivalent to the subfind halo finder. 
    
    - reciprocal_neighbours: modifies the neighbour's list so that all neighbour connections are reciprocal, i.e all particles know of all particles it is a neighbour of.
    
    - binary_search: binary search algorith returning the argument position of the largest element that is smaller than a given value.
    
    - insert_potential_sorted: inserts an particle id into a list of particle ids such that the list remains sorted in increasing order.
    
    - check: recursive watershed algorithm finding all conected particles with lower potentials segementing them into internal and surface sets which are saved internally.
    
    - check_to_sets: same as check but explicitly takes the inner and surface sets as aguments instead of using internal variables.
    
    - itt_check: itterative watershed algorithm finding all conected particles with lower potentials segementing them into internal and surface sets which are saved internally.
    
    - itt_check_to_sets: same as itt_check but explicitly takes the inner and surface sets as aguments instead of using internal variables.
    
    - sort_surface: sorts particles assigned to the surface set of the structure in order of increasing boosted potential. 
    
    - grow: grows the sturcture itteratively by including one by one the particles with the lowest bosted potential until a saddle point is found.
    
    - segment: main user function, selects all the connected particles that are assigned to the structure.
    
    '''
    def __init__(self, ngbs, pot, pos, vel, sim_params = {'scale_factor':1., 'Omega_m':1., 'Lbox':1000.}, threshold = 'EdS-cond',
                 substruct = False, no_binding = False, verbose = False):
        
        self.ngbs = ngbs
        self.pot = pot
        self.pos = pos
        self.vel = vel
        
        self.substruct = substruct
        self.verbose = verbose
        self.no_binding = no_binding
        self.threshold = threshold
        
        self.Lbox = sim_params['Lbox']
        self._Omega_m = sim_params['Omega_m']
        self._Omega_L = 1. - sim_params['Omega_m']
        self._Omega_k = 0.
        self._scale_factor = sim_params['scale_factor']
        self._H0 = 100 #In theory we should need to touch this
        self._delta_th = 0.
        self.set_delta_th(threshold)
        
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(pot.size, dtype = bool) # We may want to review these assignements
        
        self._computed = np.zeros(pot.size, dtype = bool)
        self._phi = np.zeros(pot.size, dtype = 'f8') 
        self.group = np.zeros(pot.size, dtype = 'i8')
        self.subgroup = np.zeros(pot.size, dtype = 'i8')
        self.parent = np.zeros(pot.size, dtype = 'i8')
        
        self.group_mask = np.zeros(pot.size, dtype = bool)
        self.surface_mask = np.zeros(pot.size, dtype = bool)
        self.surface_order = np.zeros(pot.size, dtype = 'i8') - 1
        
        self.subgroup_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_order = np.zeros(pot.size, dtype = 'i8') - 1
        
        self.bound_mask = np.zeros(pot.size, dtype = bool)
        
        #self._surface_size = 0
        self._current_group = 0
        self._current_subgroup = 0
        #self._lowest_pot_id = -1
        
        self.new_min = 3.4e38 # this is just initialising the variable
        self._long_range_fac = 0.5 * self._delta_th *self._Omega_m * self._scale_factor**2 * self._H0 * self._H0
        
        #self.i_in = set()
        #self.i_surf = set()
        self.debug_out = {}
        
        return
    
    # ================== Cosmology ==============================
    # Everything should be in units of h
    
    def update_cosmology(self, sim_params):
        self.Lbox = sim_params['Lbox']
        self._Omega_m = sim_params['Omega_m']
        self._Omega_L = 1. - sim_params['Omega_m']
        self._Omega_k = 0.
        self._scale_factor = sim_params['scale_factor']
        return None
    
    def H_a(self, a):
        return self._H0 * np.sqrt(self._Omega_m/a**(3)+self._Omega_k/a**(2)+self._Omega_L)
        
    def D(self, a):
        f = lambda ap: (self._Omega_m*ap**(-1.) + self._Omega_L*ap**(2) + self._Omega_k)**(-3/2)
        d, err = self.H_a(a)/self._H0 * np.array(quad(f,1e-8,a))
        D0 = self.H_a(1.)/self._H0 * np.array(quad(f,1e-8,1.))[0]
        return d/D0

    def g(self, a):
        g_i = self.D(1e-5)/1e-5
        return np.vectorize(self.D)(a)/a / g_i

    def w(self, a):
        return (self._Omega_L/self._Omega_m) * a**3

    def Omega_m(self, a):
        return self._Omega_m * a**-3 * self._H0*self._H0/self.H_a(a)/self.H_a(a) 

    def Omega_L(self, a):
        return self._Omega_L * self._H0**2/self.H_a(a)**2 

    def t_H(self):
        return 9.7779222 #h^{-1}Gyr

    def time_of_a(self, a):
        # in units of t_H
        a = np.array(a)
        if a.size < 2:
            res,err = quad(lambda ap: 1.0/(ap*self.H_a(ap)/self._H0),0,a)
            return res
        else:
            res=np.zeros_like(a)
            for i,ai in enumerate(a):
                t,err = quad(lambda ap: 1.0/(ap*self.H_a(ap)/self._H0),0,ai)
                res[i]=t
        return res
    
    def Newton_Raphson(self, f, xi, dx, tol, *args, **kwargs): 
        '''
        This is the standard Newton-Raphson root finder "borrowed" from Numerical Recipes Vol.III
        Parameters:
        ----------
        f: (callable) 1 parameter function to optmize
        xi: (float) starting position
        dx: (float) spacing used to numerically compute derivatives
        tol: (float) absolute tolerance criterion
        Outputs:
        ----------
        xi_1: (float) root of f
        '''
        verbose = self.verbose
        xi_1 = xi
        Dx = tol * 1e3
        it_num = 0
        while Dx > tol:
            xi = xi_1
            xi_1 = xi - dx * f(xi, *args)/(f(xi + dx, *args) - f(xi, *args))
            if xi_1 < 0:
                xi_1 = 0
            Dx = np.abs(xi - xi_1)
            if verbose:
                print(xi, end= ' ')
            if it_num >= 500:
                raise ValueError("The root search has not converged after 500 itterations")
        if verbose:
            print("root: ", xi_1, end= '\n')
        return xi_1
    
    def get_zeta(self, a):
        def func(zeta, a):
            t_max = self.time_of_a(a)
            dy = lambda x: 1/np.sqrt(1/x - 1 + zeta * (x*x -1))
            y,err = quad(dy, 0, 1)
            return  t_max - np.sqrt(zeta/self._Omega_L) * y

        zeta = self.Newton_Raphson(f = lambda z: func(z,a), xi = 0.1, dx = 1e-11, tol = 1e-10)
        return zeta
    
    def set_delta_th(self, threshold):
        if 'EdS' in threshold:
            if "cond" in threshold:
                self._delta_th = 0.
            elif "coll" in threshold:
                self._delta_th = 3/5*(3*np.pi/2)**(1/3.)
            elif "ta-lin" in threshold:
                self._delta_th = 3/5*(3*np.pi/4)**(1/3.)
            elif "ta-eul" in threshold:
                self._delta_th = 9*np.pi**2/16 - 1
            else:
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta", "LCDM-cond", "LCDM-coll", and "LCDM-ta"')
        elif 'LCDM' in threshold:
            self._zeta = self.get_zeta(self._scale_factor) 
            if "cond" in threshold:
                self._delta_th = 9/10 * (2 * self.w(self._scale_factor))**(1/3)
            elif "coll" in threshold:
                self._delta_th = 3/5 * self.g(self._scale_factor) * (1 + self._zeta) * (self.w(self._scale_factor)/self._zeta)**(1./3.)
            elif "ta-lin" in threshold:
                t_c = self.time_of_a(self._scale_factor)
                t_ta = t_c/2
                a_ta = self.Newton_Raphson(f = lambda a: self.time_of_a(a) - t_ta, xi = 0.1, dx = 1e-6, tol = 1e-6)
                zeta_ta = self.get_zeta(a_ta)
                self._delta_th = 3/5 * self.g(a_ta) * (1 + zeta_ta) * (self.w(a_ta)/zeta_ta)**(1./3.)
            elif "ta-eul" in threshold:
                self._delta_th = self._Omega_L/self._Omega_m * self._scale_factor**3/self._zeta - 1
            else:
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta", "LCDM-cond", "LCDM-coll", and "LCDM-ta"')
        else:
            raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta", "LCDM-cond", "LCDM-coll", and "LCDM-ta"')
        return None
    
    

    
    # ================== Utility Methods ========================
    
    #####=================== Boost!!! ===========================
    def set_x0(self, x0):
        '''
        Function which sets 'x0' the reference position for the calculation of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference position vector.

        '''
        self.x0 = x0
        # Reset the cache values
        self._computed = np.zeros(self.pot.size, dtype = bool)
        self._phi = np.zeros(self.pot.size, dtype = 'f8') 
        return
    
    def set_acc0(self, acc0):
        '''
        Function which sets 'acc0' the reference acceleration for the calculation of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.

        '''
        self.acc0 = acc0
        # Reset the cache values
        self._computed = np.zeros(self.pot.size, dtype = bool)
        self._phi = np.zeros(self.pot.size, dtype = 'f8') 
        return
    
    def recentre_positions(self, pos, x0):
        '''
        Function which recentres and wraps the position vector 'pos' to a new centre at 'x0'.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.
        pos: (array of d-floats) positions of of particles in d-dimensional space.

        '''
        return (pos -x0 - self.Lbox/2)%self.Lbox - self.Lbox/2
    
    def phi_boost(self, i):
        '''
        Function which calculates the boosted potential at the position of particle i
        
        Parameters:
        ----------
        i: (int or array of ints) index/array of indices of the particles for which to calculate the boosted potential.
        
        Returns:
        ----------
        phi_boost: (float or array of floats) boosted potential of the requested particles
        '''
        
        if self.x0 is None:
            raise ValueError('A reference position x0 must be set first. This can be done by calling the set_x0 or segment methods.')
        if self.acc0 is None:
            raise ValueError('A reference acceleration acc0 must be set first. This can be done by calling the set_acc0 or segment methods.')
        x = self.recentre_positions(self.pos[i], self.x0)
        if np.all(self._computed[i]):
            return self._phi[i]
        else:
            if np.array(i).size > 1:
                self._computed[i] = True
                self._phi[i] = self.pot[i] + np.sum(x * np.array(self.acc0), axis = 1) - self._long_range_fac * np.sum(x*x, axis = 1)
                return self._phi[i]
            else:
                self._computed[i] = True
                self._phi[i] = self.pot[i] + np.sum(x * np.array(self.acc0)) - self._long_range_fac * np.sum(x*x)
                return self._phi[i]
    
    #####=================== Neighbours ==================
    
    def subfind_neighbours(self):
        '''
        Function altering the list of neighbours to obtain a connectivity equivalent to that used within the subfind algorithm. See Springel et al. 2001, MNRAS 328, 726–750 for details on the subfind algorithm. Warning this functgion will rewrite the entire neighbours list and as such can take an extremely long amount of time if used on too many particles.
        
        Parameters:
        ----------
        None
        '''
        
        recip_ngbs = list(self.ngbs)
        
        for i in tqdm(range(self.pos.shape[0])):
            cond = self.phi_boost(recip_ngbs[i]) < self.phi_boost(i)
            recip_ngbs[i] = np.array(recip_ngbs[i][cond][:2])
    
        for i in tqdm(range(self.pos.shape[0])):
            for j in recip_ngbs[i]:
                if i not in set(recip_ngbs[j]):
                    recip_ngbs[j] = np.array(list(recip_ngbs[j]) + [i,])
        self.ngbs = recip_ngbs
        return
    
    def reciprocal_neighbours(self):
        '''
        Function altering the list of neighbours to obtain a reciprocal connectivity, i.e if a particle is a neighbour of another then it is also connected to that other particle. Warning this functgion will rewrite the entire neighbours list and as such can take an extremely long amount of time if used on too many particles.
        
        Parameters:
        ----------
        None
        '''
        
        recip_ngbs = list(self.ngbs)
        
        for i in tqdm(range(self.pos.shape[0])):
            for j in recip_ngbs[i]:
                if i not in set(recip_ngbs[j]):
                    recip_ngbs[j] = np.array(list(recip_ngbs[j]) + [i,])
        self.ngbs = recip_ngbs
        return
    
    #####=================== Sorting ==================
    
    def binary_search(self, element, l):
        '''
        List binary argument search fuction. Finds the argument in the sorted array l which would be just below 'element' such that 'l[:arg] + [element,] + l[arg:]' would be a sorted list
        
        Parameters:
        ----------
        element: element we want to add to the array
        l: array sorted in increasing order
        
        Returns:
        ----------
        arg: (int) Argument of the largest element of l that is below 'element'
        '''
        return stl.binary_search(element, l)
   
    def unmix(self, a, order):
        #a_new = np.zeros(a.shape, dtype = a.dtype)
        #for i,j in enumerate(order):
        #    a_new[j] = a[i]
        return stl.unmix(a, order)#a_new

    def is_sorted(self, a):
        '''
        Function checking if array a is sorted in ascending order
        
        Parameters:
        ----------
        a: (array of float) array to check
        
        Returns:
        ----------
        res: (bool) True is the array is ascending
        '''
        return np.all(np.diff(a) >= 0)
    
    def sort_surface(self, surface_mask):
        '''
        Function which sorts the particles in the set 'i_surf' in order of increasing potential
        
        Parameters:
        ----------
        i_surf: (set of ints) set of particles which are direct neighbours of particles that are within the structure.
        
        Returns:
        ----------
        id_surf: (array of ints) array of the sorted indices
        phi_surf: (array of floats) array of boosted potentials in the order of the indices
        '''
        
        id_surf = np.where(surface_mask)[0]
        phi_surf = self.phi_boost(id_surf)
        order = np.argsort(phi_surf)
        new_order = np.zeros(order.size, dtype = order.dtype)
        for i,j in enumerate(order):
            new_order[j] = i
        return id_surf, new_order

        
    def insert_potential_sorted(self, i, l, phi_arr):
        '''
        Function inserting the particle index i into the sorted list of indices l according the value of the boosted potetial. In the current implementation this function takes the array of boosted potentials to avoid needing to recompute it.
        
        Parameters:
        ----------
        i: (int) index of the particle to add to the sorted list
        l: (array of int) list of particle indices according to their bosted potential
        phi_arr: (array of floats) list of boosted potentials of the particles in l
        
        Returns:
        ----------
        l: (array of int) updated list of particle indices according to their bosted potential
        phi_arr: (array of floats) updated list of boosted potentials of the particles in l
        '''
        
        
        if len(phi_arr) == 0:
            l = np.array([i,], dtype = 'i4')
            phi_arr = np.array([self.phi_boost(i),], dtype = 'f8')
            return l, phi_arr
        #if not self.is_sorted(phi_arr): phi_arr = np.sort(phi_arr) # This is a temporary fix... which slows the code a lot. Have to find where the ordering is lost
        arg = self.binary_search(self.phi_boost(i), phi_arr)
        l = np.array(list(l[:arg]) + [i,] + list(l[arg:]))
        phi_arr = np.array(list(phi_arr[:arg]) + [self.phi_boost(i),] + list(phi_arr[arg:]))
        return l, phi_arr
    
    def insert_mask_potential_sorted(self, i, order, mask):
        '''
        Function inserting the particle index i into the sorted list of indices l according the value of the boosted potetial. In the current implementation this function takes the array of boosted potentials to avoid needing to recompute it.
        
        Parameters:
        ----------
        i: (int) index of the particle to add to the sorted list
        l: (array of int) list of particle indices according to their bosted potential
        phi_arr: (array of floats) list of boosted potentials of the particles in l
        
        Returns:
        ----------
        l: (array of int) updated list of particle indices according to their bosted potential
        phi_arr: (array of floats) updated list of boosted potentials of the particles in l
        '''
        
        phi_arr = self.phi_boost(self.unmix(np.where(mask)[0],order[mask]))
        
        if len(phi_arr) == 0:
            order[i] = 1
            return order, mask
        #if not self.is_sorted(phi_arr): phi_arr = np.sort(phi_arr) # This is a temporary fix... which slows the code a lot. Have to find where the ordering is lost
        arg = self.binary_search(self.phi_boost(i), phi_arr)
        order[order >= arg] += 1
        order[i] = arg
        mask[i] = True
        return order, mask
    
    def remove_mask_potential_sorted(self, i, order, mask):
        '''
        Function inserting the particle index i into the sorted list of indices l according the value of the boosted potetial. In the current implementation this function takes the array of boosted potentials to avoid needing to recompute it.
        
        Parameters:
        ----------
        i: (int) index of the particle to add to the sorted list
        l: (array of int) list of particle indices according to their bosted potential
        phi_arr: (array of floats) list of boosted potentials of the particles in l
        
        Returns:
        ----------
        l: (array of int) updated list of particle indices according to their bosted potential
        phi_arr: (array of floats) updated list of boosted potentials of the particles in l
        '''
        
        
        #if not self.is_sorted(phi_arr): phi_arr = np.sort(phi_arr) # This is a temporary fix... which slows the code a lot. Have to find where the ordering is lost
        arg = order[i]
        order[order > arg] -= 1
        order[i] = -1
        mask[i] = False
        return order, mask
    
    # ================== Particle Assignment ====================
    
    def first_minimum(self, i0, r = 1):
        counter = 1
        i = i0
        x0 = self.pos[i0]
        phi_ngbs = self.phi_boost(self.ngbs[i])
        self.debug_out["first_min"] = i
        while np.min(phi_ngbs) < self.phi_boost(i):
            counter += 1
            self.debug_out["first_min"] = self.ngbs[i][np.argmin(phi_ngbs)]
            i = self.ngbs[i][np.argmin(phi_ngbs)]
            phi_ngbs = self.phi_boost(self.ngbs[i])
            if counter % 100 == 0:
                x = self.recentre_positions(self.pos[i], x0)
                dist = np.sqrt(np.sum(x*x))
                if dist > r:
                    raise RecursionError(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                if counter > 1000:
                    raise RecursionError("The minimum was not found after 1000 steps.\nThis may indicate that there is either no miminum or that you are starting index is too far away.")
        x = self.recentre_positions(self.pos[i], x0)
        dist = np.sqrt(np.sum(x*x))
        if dist > r:
            raise RecursionError(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                  \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
        return i
    
    def itt_check(self, i):
        '''
        Itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle i. 
        
        Note that the particles are saved within a set 'self.i_in' that is saved internally to the class and all the direct neighbours to the particles in this set are saved to a second set 'self.i_surf'. 
        
        These sets should not be modified directly as this can cause issues with the segmentation algorithm.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        '''
        self.group[i] = self._current_group
        self.group_mask[i] = True
        self.parent[i] = self._current_group
        self.visited[i] = True

        # Calculate it's potential and define its neighbours as surface particles
        phi_curr = self.phi_boost(i)
        self.surface_mask[self.ngbs[i]] = True
        # Sort the surface
        id_surf, order = self.sort_surface(self.surface_mask) # This function outputs a numpy array

        self.surface_order[self.surface_mask] = order
        
        # Take surface particles with a potential lower that current
        while np.any(self.phi_boost(np.where(self.surface_mask)[0]) - phi_curr < 0):
            j = self.unmix(np.where(self.surface_mask)[0],self.surface_order[self.surface_mask])[0]
            self.group[j] = self._current_group
            self.group_mask[j] = True
            self.parent[j] = self._current_group

            self.visited[j] = True

            self.surface_order, self.surface_mask = self.remove_mask_potential_sorted(j, self.surface_order, self.surface_mask)
            for k in self.ngbs[j]:
                if self.surface_mask[k] or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    self.surface_order, self.surface_mask = self.insert_mask_potential_sorted(k, self.surface_order, self.surface_mask)
        return
    

    
    def itt_check_to_sets(self, i, phi_min):
        '''
        itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle k. 
        
        Note that this function is similar to check(i,k), but has the major difference of explicitly taking the sets of indices 'i_in' and 'i_surf' as arguments. 
        This allows to probe newly found potential wells without altering the sets of particles currently assigned to the man structure or its surface.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        i_in: (set of ints) set of indices of particles curently assigned to a structure
        i_surf: (set of ints) set of particles which are direct neighbours of particles that are within the structure.
        phi_min: (float) current minimum of the potential well
        '''
        # Start by adding the refference particle to the group
        self.subgroup_mask[i] = True
        
        self.visited[i] = True
        # Calculate it's potential and define its neighbours as surface particles
        phi_curr = self.phi_boost(i)
        self.subsurface_mask[self.ngbs[i][~self.visited[self.ngbs[i]]]] = True
        # Sort the surface
        id_surf, order = self.sort_surface(self.subsurface_mask) # This function outputs a numpy array

        self.subsurface_order[self.subsurface_mask] = order
        
        # Take surface particles with a potential lower that current
        
        counter = 1
        r = 3 * np.max(self.recentre_positions(self.pos[self.group_mask], self.x0))
        while np.any(self.phi_boost(np.where(self.subsurface_mask)[0]) - phi_curr < 0):
            counter += 1
            
            j = self.unmix(np.where(self.subsurface_mask)[0],self.subsurface_order[self.subsurface_mask])[0]
            # By taking this one we ensure we are always going down the lowest brach first.
            phi_j = self.phi_boost(j)
            
            if counter % 100 == 0: # Check if we haven't wandered too far away
                x = self.recentre_positions(self.pos[j], self.x0)
                dist = np.sqrt(np.sum(x*x))
                if dist > r:
                    raise RecursionError(f"The new minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                if np.sum(self.subgroup_mask) > 4*np.sum(self.group_mask):
                    raise RecursionError(f"Trying to add in a structure which is much larger than the current structure. Exiting.")
                    
            # If we are in this we already know that j has a lower potential
            # So we can add it directly and remove it from the surface
            self.subsurface_order, self.subsurface_mask = self.remove_mask_potential_sorted(j, self.subsurface_order, self.subsurface_mask)
            if self.visited[j]:
                continue
            self.subgroup_mask[j] = True
            self.visited[j] = True
            if phi_j < phi_min: # Check to see if we are going below the current minimum potential
                self.new_min = phi_j
                return
            # Add neighbours to surface
            for k in self.ngbs[j]:
                if self.subsurface_mask[k] or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    self.subsurface_order, self.subsurface_mask = self.insert_mask_potential_sorted(k, self.subsurface_order, self.subsurface_mask)
            # Take surface particles with a potential lower that current
        return
    


    def grow(self):
        '''
        Itterative algorithm which grows the structure one particle at a time. At each step the particle with the lowest potential on the surface is considered. If all of its neighbours with lower potential currently belongs to the structure then it is also added to the structure. If any of its lower neighbours have lower potentials and are NOT associated to the structure then the particle is marked as a saddle point. These braches are then explored with a watershed algorithm to find if any brach has reaches values that are deeper than the current minimum of the boosted potential. If so the algorithm terminates.
        
        Parameters:
        ----------
        i_in: (set of ints) set of indices of particles curently assigned to a structure
        i_surf: (set of ints) set of particles which are direct neighbours of particles that are within the structure
        
        Returns:
        ----------
        i_min: (int) index of the particle with the lowest boosted potential in the group
        i_sad: (int) index of the particle corresponding to the saddle point in the boosted potetnial
        n_part: (int) number of particles assigned to that group
        (optional outputs)
        subs: (list of 3-ints) same properties as for the main output but given for all thesubgroups that have been found
        i_in: (set of ints) set of indices of all the particles contained in the structure
        '''
        
        phi_min = np.min(self.phi_boost(np.where(self.group_mask)[0]))
    
        i_prev = -1
        id_surf, order = self.sort_surface(self.surface_mask)
        self.surface_order[self.surface_mask] = order
        
        for _ in range(len(self.pot)): # <== I just don't want to write 'while True:'
            
            if not np.any(self.surface_mask):
                if self.verbose: print('Surface empty', flush = True)
                self.debug_out["exit"] = 'SE'
                break

            i_cons = self.unmix(np.where(self.surface_mask)[0],self.surface_order[self.surface_mask])[0]
            
            if i_cons == i_prev:
                raise RecursionError(f"The algorithm seems to have gotten stuck at i = {i_cons} with {len(self.group[self.group_mask])} particles assinged and {np.where(self.surface_mask)[0]} pending...")
            i_prev = i_cons
            #len_in = len(i_in); len_surf = len(i_surf)
            self.visited[i_cons] = True

            phi_boost_ngbs = self.phi_boost(self.ngbs[i_cons])
            low_pot_cond = (phi_boost_ngbs < self.phi_boost(i_cons))
            ilow = self.ngbs[i_cons][low_pot_cond]
                   
            if len(ilow) == 0:
                #This can happen when hitting the boundaries
                self.surface_order, self.surface_mask = self.remove_mask_potential_sorted(i_cons, self.surface_order, self.surface_mask)
                continue
                
            i_in = np.where(self.group_mask)[0]
            if_cond = np.zeros(ilow.size, dtype = bool)
            for k,elem in enumerate(ilow):
                if_cond[k] = (elem == i_in[self.binary_search(elem, i_in)-1])
            
            if np.all(if_cond):
                # Tag i_cons as part of the main group
                self.group[i_cons] = self._current_group
                self.group_mask[i_cons] = True
                self.parent[i_cons] = self._current_group
        
                self.surface_order, self.surface_mask = self.remove_mask_potential_sorted(i_cons, self.surface_order, self.surface_mask)
                for k in self.ngbs[i_cons]:
                    if self.surface_mask[k] or self.visited[k]:
                        # Avoid duplicates or going back to a particle that has already been visited
                        continue
                    else: 
                        # Insert particle in sorted order
                        self.surface_order, self.surface_mask = self.insert_mask_potential_sorted(k, self.surface_order, self.surface_mask)
                

            else:
                
                self.visited[i_cons] = False
                self.new_min = self.phi_boost(i_cons)
                
                self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)
                self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
                self.subsurface_order = np.zeros(self.pot.size, dtype = 'i8') - 1
                
                try:
                    self.itt_check_to_sets(i_cons, phi_min)
                except RecursionError:
                    if self.verbose: print('Traveled too far exitting... ', np.sum(self.subgroup_mask), end = ' ', flush = True)
                    self.debug_out["exit"] = 'TF'
                    self.visited[self.subgroup_mask | self.subsurface_mask] = False
                    break
                
                if self.new_min <= phi_min:
                    # Moved into a lower potential well => exit
                    if self.verbose: print(f'Found lower minimum ({i_cons}):', np.sum(self.subgroup_mask), end = ' ', flush = True)
                    self.debug_out["exit"] = 'LM'
                    self.visited[self.subgroup_mask | self.subsurface_mask] = False
                    self.debug_out["i_in_new"] = np.where(self.subgroup_mask)[0]
                    self.debug_out["i_surf_new"] = np.where(self.subsurface_mask)[0]
                    break
                    
                
                else:
                    # Found a structure => add it in 
                    
                    self.visited[i_cons] = True #remove i_cons
                    for k in np.where(self.subsurface_mask)[0]:
                        if self.surface_mask[k] or self.visited[k]:
                            # Avoid duplicates or going back to a particle that has already been visited
                            continue
                        else: 
                            # Insert particle in sorted order
                            self.surface_order, self.surface_mask = self.insert_mask_potential_sorted(k, self.surface_order, self.surface_mask)
                    self.group[self.subgroup_mask] = self._current_group
                    self.group_mask[self.subgroup_mask] = True
                    self.parent[self.subgroup_mask] = self._current_group
                    
                    if self.surface_mask[i_cons]:
                        self.surface_order, self.surface_mask = self.remove_mask_potential_sorted(i_cons, self.surface_order, self.surface_mask)

                    
                    phi_min = np.min(self.phi_boost(np.where(self.group_mask)[0]))
                    if np.sum(self.subgroup_mask) > 20:
                        self._current_subgroup += 1
                        self.subgroup[self.subgroup_mask] = self._current_subgroup
                    
                if np.sum(self.group_mask) > 0.25*self.pot.size:
                    # Temporary measure to get a catalogue in EdS-cond
                    if self.verbose: print('Reached size limit:', np.sum(self.group_mask), end = ' ', flush = True)
                    self.debug_out["exit"] = 'SZ'
                    break
        
        ids = np.where(self.group_mask)[0]
        i_sad = ids[np.argmax(self.phi_boost(ids))]
        i_min = ids[np.argmin(self.phi_boost(ids))]      
        return i_min, i_sad
    
    def is_bound(self, i_min, i_sad):
        '''
        Binding check. compares the mechanical energy Kinetic + boosted Potential of all particles in i_in to the boosted potential evajulated at the location of i_sad, representing a saddle point.
        
        Parameters:
        ----------
        i_in: (set of N int) indices of connected particles with boosted potential below the saddle point energy.
        i_sad: (int) particle id of the saddle point particle
        
        Returns:
        ----------
        mask: (array of N bool) mask selecting only the particles that are bound to the structure.
        '''
        i_in_arr = np.where(self.group_mask)[0]
        v_mean = np.median(self.vel[i_in_arr], axis = 0) # remove the mean velocity of the group
        v_in = self.vel[i_in_arr] - v_mean
        
        x = self.recentre_positions(self.pos[i_in_arr],self.pos[i_min])
        x2 = np.sum(x*x, axis = 1) 
        K = 0.5 * np.sum(v_in * v_in, axis = 1) + self._scale_factor * self.H_a(self._scale_factor) * np.sum(x*v_in, axis = 1) 
        phi_p = self._scale_factor**2 * self.phi_boost(i_in_arr) + 0.5*(self._Omega_m/2 + 1)*self._H0**2* self._scale_factor**-1 * x2
        phi_p_sad = self._scale_factor**2 * self.phi_boost(i_sad) + 0.5*(self._Omega_m/2 + 1)*self._H0**2* self._scale_factor**-1 * x2
        E = K + phi_p # <= Converted to physical potential
        bound = E < phi_p_sad
        self.bound_mask[i_in_arr[bound]] = True
        return self.bound_mask

    
    def segment(self, i0, acc0, r = 1):
        '''
        Main user function. Segments all particles belong to the same group as particle 'i0'. The boosted potential here is calculated with respect to the position of particle 'i0' and with the acceleration vector 'acc0'
        
        Parameters:
        ----------
        i0: (int) index of starting particle
        acc0: (array of floats) acceleration vector used to calculate the boosted potential
        
        Returns:
        ----------
        i_min: (int) index of the particle with the lowest boosted potential in the group
        i_sad: (int) index of the particle corresponding to the saddle point in the boosted potetnial
        n_part: (int) number of particles assigned to that group
        (optional outputs)
        subs: (list of 3-ints) same properties as for the main output but given for all thesubgroups that have been found
        i_in: (set of ints) set of indices of all the particles contained in the structure
        '''
        self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        self.acc0 = acc0
        self.x0 = self.pos[i0]
        
        self._current_group += 1
        self._current_subgroup = 0
        # Verify that the minimum is not too far away.
        i0 = self.first_minimum(i0, r)
        # Find all particles with potential lower than minimum (This should only give back 1 particle)
        self.itt_check(i0)
        # Grow potential surface
        i_min, i_sad = self.grow()
        self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        #Binding check
        if self.no_binding: 
            return np.where(self.group_mask)[0]
        self.bound_mask = self.is_bound(i_min, i_sad)
        return np.where(self.group_mask & self.bound_mask)[0] # output: i_min, i_sad, n_part(, subs, i_in)
    
    def reset(self):
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(self.pot.size, dtype = bool) # We may want to review these assignements
        
        self._computed = np.zeros(self.pot.size, dtype = bool)
        self._phi = np.zeros(self.pot.size, dtype = 'f8') 
        self.group = np.zeros(self.pot.size, dtype = 'i8')
        self.subgroup = np.zeros(self.pot.size, dtype = 'i8')
        self.parent = np.zeros(self.pot.size, dtype = 'i8')
        
        self.group_mask = np.zeros(self.pot.size, dtype = bool)
        self.surface_mask = np.zeros(self.pot.size, dtype = bool)
        self.surface_order = np.zeros(self.pot.size, dtype = 'i8') - 1
        
        self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_order = np.zeros(self.pot.size, dtype = 'i8') - 1
        
        self.bound_mask = np.zeros(self.pot.size, dtype = bool)
        return None