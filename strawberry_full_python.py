import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

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
                 substruct = False, itt=True, no_binding = False, verbose = False):
        
        self.ngbs = ngbs
        self.pot = pot
        self.pos = pos
        self.vel = vel
        
        self.substruct = substruct
        self.verbose = verbose
        self.no_binding = no_binding
        self.itt = itt
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
        self.visited = np.zeros(pot.size, dtype = bool) # We may want to review this assignement
        self.new_min = 3.4e38 # this is just initialising the variable
        # In spherical symetry to get threshold at <delta> we need delta' = 2pi*<delta> - 1
        self._long_range_fac = 0.5 * (1 + (2*np.pi*self._delta_th - 1)) *self._Omega_m * self._scale_factor**2 * self._H0 * self._H0
        
        self.i_in = set()
        self.i_surf = set()
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
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-eul", "EdS-ta-lin", "LCDM-cond", "LCDM-coll", and "LCDM-ta-eul", "LCDM-ta-lin"')
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
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-eul", "EdS-ta-lin", "LCDM-cond", "LCDM-coll", and "LCDM-ta-eul", "LCDM-ta-lin"')
        else:
            raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-eul", "EdS-ta-lin", "LCDM-cond", "LCDM-coll", and "LCDM-ta-eul", "LCDM-ta-lin"')
        return None
    
    
    # ================== Particle Assignment ====================
    def set_x0(self, x0):
        '''
        Function which sets 'x0' the reference position for the calculation of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference position vector.

        '''
        self.x0 = x0
        return
    
    def set_acc0(self, acc0):
        '''
        Function which sets 'acc0' the reference acceleration for the calculation of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.

        '''
        self.acc0 = acc0
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
        if np.array(i).size > 1:
            return self.pot[i] + np.sum(x * np.array(self.acc0), axis = 1) - self._long_range_fac * np.sum(x*x, axis = 1)
        else:
            return self.pot[i] + np.sum(x * np.array(self.acc0)) - self._long_range_fac * np.sum(x*x)

    def check(self, i, k):
        '''
        Recursive watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle k. 
        
        Note that the particles are saved within a set 'self.i_in' that is saved internally to the class and all the direct neighbours to the particles in this set are saved to a second set 'self.i_surf'. 
        
        These sets should not be modified directly as this can cause issues with the segmentation algorithm.
        
        Parameters:
        ----------
        i: (int) index of the particle to being considered.
        k: (int) index of the particle defining the level of the potential
        '''
        
        if self.visited[i]: return

        self.visited[i] = True
        if self.phi_boost(i) - self.phi_boost(k) <= 0:
            self.i_in.add(i)
            for j in self.ngbs[i]:
                self.check(j,k)
            return
        else:
            self.i_surf.add(i)
            return
    
    def check_to_sets(self, i, k, i_in, i_surf, phi_min):
        '''
        Recursive watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle k. 
        
        Note that this function is similar to check(i,k), but has the major difference of explicitly taking the sets of indices 'i_in' and 'i_surf' as arguments. This allows to probe newly found potential wells without altering the sets of particles currently assigned to the man structure or its surface.
        
        Parameters:
        ----------
        i: (int) index of the particle to being considered.
        k: (int) index of the particle defining the level of the potential
        i_in: (set of ints) set of indices of particles curently assigned to a structure
        i_surf: (set of ints) set of particles which are direct neighbours of particles that are within the structure.
        phi_min: (float) current minimum of the potential well
        '''
        if self.new_min < phi_min:
            return

        if self.visited[i]: 
            return

        self.visited[i] = True
        if self.phi_boost(i) - self.phi_boost(k) <= 0:
            i_in.add(i)
            if self.phi_boost(i) < phi_min:
                self.new_min = self.phi_boost(i)
                return
            ngbs_order = np.argsort(self.phi_boost(self.ngbs[i]))
            for j in self.ngbs[i][ngbs_order]:
                self.check_to_sets(j, k, i_in, i_surf, phi_min)
            return
        else:
            i_surf.add(i)
            return
    
    
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
        if not self.is_sorted(l): raise ValueError("The list provided is not sorted in increasing order.")
        arg = np.arange(len(l))
        while len(l) > 1:
            N = len(l)
            if l[N//2] > element:
                l = l[:N//2]
                arg = arg[:N//2]
            else:
                l = l[N//2:]
                arg = arg[N//2:]
        arg += 1
        if arg == 1 and element < l:
            arg = np.array([0])
        return arg[0]
        
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
            l = np.array([i,], dtype = 'i8')
            phi_arr = np.array([self.phi_boost(i),], dtype = 'f8')
            return l, phi_arr
        #if not self.is_sorted(phi_arr): phi_arr = np.sort(phi_arr) # This is a temporary fix... which slows the code a lot. Have to find where the ordering is lost
        arg = self.binary_search(self.phi_boost(i), phi_arr)
        l = np.array(list(l[:arg]) + [i,] + list(l[arg:]))
        phi_arr = np.array(list(phi_arr[:arg]) + [self.phi_boost(i),] + list(phi_arr[arg:]))
        return l, phi_arr
        
    def itt_check(self, i):
        '''
        Itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle i. 
        
        Note that the particles are saved within a set 'self.i_in' that is saved internally to the class and all the direct neighbours to the particles in this set are saved to a second set 'self.i_surf'. 
        
        These sets should not be modified directly as this can cause issues with the segmentation algorithm.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        '''
        # Start by adding the refference particle to the group
        self.i_in.add(i)
        self.visited[i] = True
        # Calculate it's potential and define its neighbours as surface particles
        phi_curr = self.phi_boost(i)
        t_surf = self.ngbs[i]
        # Sort the surface
        t_surf, phi_t_surf = self.sort_surface(t_surf) # This function outputs a numpy array
        # Take surface particles with a potential lower that current
        t_low = t_surf[phi_t_surf - phi_curr < 0]
        while len(t_low) > 0:
            j = t_surf[0] # By taking this one we ensure we are always going down the lowest brach first.
            # If we are in this we already know that j has a lower potential
            # So we can add it directly an remove it from the surface
            self.i_in.add(j)
            t_surf = t_surf[1:]
            phi_t_surf = phi_t_surf[1:] 
            self.visited[j] = True
            # Add neighbours to surface
            for k in self.ngbs[j]:
                if k in set(t_surf) or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    t_surf, phi_t_surf = self.insert_potential_sorted(k, t_surf, phi_t_surf)
            # Take surface particles with a potential lower that current
            t_low = t_surf[phi_t_surf - phi_curr < 0]
        # Save the surface set needed for the grow function
        self.i_surf = set(t_surf)
        return
    
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
    
    def itt_check_to_sets(self, i, i_in, i_surf, phi_min):
        '''
        itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle k. 
        
        Note that this function is similar to check(i,k), but has the major difference of explicitly taking the sets of indices 'i_in' and 'i_surf' as arguments. This allows to probe newly found potential wells without altering the sets of particles currently assigned to the man structure or its surface.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        i_in: (set of ints) set of indices of particles curently assigned to a structure
        i_surf: (set of ints) set of particles which are direct neighbours of particles that are within the structure.
        phi_min: (float) current minimum of the potential well
        '''
        # Start by adding the refference particle to the group
        i_in.add(i)
        self.visited[i] = True
        # Calculate it's potential and define its neighbours as surface particles
        phi_curr = self.phi_boost(i)
        visit = self.visited[self.ngbs[i]]
        t_surf = self.ngbs[i][~visit]
        # Sort the surface
        t_surf, phi_t_surf = self.sort_surface(t_surf) # This function outputs numpy arrays
        # Take surface particles with a potential lower that current
        t_low = t_surf[phi_t_surf <= phi_curr]
        counter = 1
        r = 3 * np.max(self.recentre_positions(self.pos[list(self.i_in)], self.x0))
        while len(t_low) > 0:
            counter += 1
            
            j = t_surf[0] # By taking this one we ensure we are always going down the lowest brach first.
            phi_j = phi_t_surf[0]
            
            if counter % 100 == 0: # Check if we haven't wandered too far away
                x = self.recentre_positions(self.pos[i], self.x0)
                dist = np.sqrt(np.sum(x*x))
                if dist > r:
                    raise RecursionError(f"The new minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                if len(i_in) > 4*len(self.i_in):
                    raise RecursionError(f"Trying to add in a structure which is much larger than the current structure. Exiting.")
                    
            # If we are in this we already know that j has a lower potential
            # So we can add it directly and remove it from the surface
            t_surf = t_surf[1:]
            phi_t_surf = phi_t_surf[1:]
            if self.visited[j]:
                continue
            i_in.add(j)
            self.visited[j] = True
            if phi_j < phi_min: # Check to see if we are going below the current minimum potential
                self.new_min = phi_j
                return
            # Add neighbours to surface
            for k in self.ngbs[j]:
                if k in set(t_surf) or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else:
                    # Insert particle in sorted order
                    t_surf, phi_t_surf = self.insert_potential_sorted(k, t_surf, phi_t_surf)
                    #if not self.is_sorted(phi_t_surf): raise ValueError("The list provided is not sorted in increasing order.")
            # Take surface particles with a potential lower that current
            
            t_low = t_surf[phi_t_surf <= phi_curr]
        # Save the surface set needed for the grow function
        i_surf |= set(t_surf)
        return
    
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
    
    def sort_surface(self, i_surf):
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
        
        id_surf = np.array(list(i_surf), dtype = np.int64)
        phi_surf = self.phi_boost(id_surf)
        order = np.argsort(phi_surf)
        id_surf = id_surf[order]
        phi_surf = phi_surf[order]
        return id_surf, phi_surf

    


    def grow(self, i_in, i_surf):
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
        phi_min = np.min(self.phi_boost(list(i_in)))
        #while True:
        subs = []
        len_in = 0
        len_surf = 0
        i_prev = -1
        for _ in range(4*len(self.pot)): # <== I just don't want to write 'while True:'
            
            id_surf, phi_surf = self.sort_surface(i_surf)
            
            if len(i_surf) == 0:
                if self.verbose: print('Surface empty', flush = True)
                self.debug_out["exit"] = 'SE'
                break
                
            i_cons = id_surf[0]
            if len_in == len(i_in) and len_surf == len(i_surf) and i_cons == i_prev:
                raise RecursionError(f"The algorithm seems to have gotten stuck at i = {i_cons} with {len(i_in)} particles assinged and {len(i_surf)} pending...")
            i_prev = i_cons
            len_in = len(i_in); len_surf = len(i_surf)
            self.visited[i_cons] = True

            phi_boost_ngbs = self.phi_boost(self.ngbs[i_cons])
            low_pot_cond = (phi_boost_ngbs < phi_surf[0])
            ilow = set(self.ngbs[i_cons][low_pot_cond])

            if len(ilow) == 0:
                #This can happen when hitting the boundaries
                i_surf.remove(i_cons)
                continue

            if ilow & i_in == ilow:
                i_in.add(i_cons)
                cons_ngbs = self.ngbs[i_cons][~low_pot_cond]
                i_surf |= set(cons_ngbs[~self.visited[cons_ngbs]])
                i_surf.remove(i_cons)

            else:
                i_in_new = set()
                i_surf_new = set()
                self.visited[i_cons] = False
                self.new_min = self.phi_boost(i_cons)
                
                # Try and fill in a subhalo
                if self.itt:
                    try:
                        self.itt_check_to_sets(i_cons, i_in_new, i_surf_new, phi_min)
                    except RecursionError:
                        if self.verbose: print('Traveled too far exitting... ', len(i_in_new), end = ' ', flush = True)
                        self.debug_out["exit"] = 'TF'
                        self.visited[list(i_in_new | i_surf_new)] = False
                        break
                else:    
                    try:
                        self.check_to_sets(i_cons, i_cons, i_in_new, i_surf_new, phi_min)
                    except RecursionError:
                        if self.verbose: print('Recursion too deep exitting... ', len(i_in_new), end = ' ', flush = True)
                        self.debug_out["exit"] = 'RE'
                        self.visited[list(i_in_new | i_surf_new)] = False
                        break
                
                if self.new_min <= phi_min:
                    # Moved into a lower potential well => exit
                    if self.verbose: print('Found lower minimum:', len(i_in_new), end = ' ', flush = True)
                    self.debug_out["exit"] = 'LM'
                    self.visited[list(i_in_new | i_surf_new)] = False
                    self.debug_out["i_in_new"] = i_in_new
                    self.debug_out["i_surf_new"] = i_surf_new
                    break
                else:
                    # Found a structure => add it in 
                    i_in |= i_in_new
                    i_surf |= i_surf_new
                    i_surf.remove(i_cons)
                    phi_min = np.min(self.phi_boost(list(i_in)))
                    if len(i_in_new) > 20:
                        ids = list(i_in_new)
                        phi_loc = self.phi_boost(ids)
                        new_sub = {}
                        new_sub["i_min"] = ids[np.argmin(phi_loc)]; new_sub["i_sad"] = ids[np.argmax(phi_loc)]
                        new_sub["n_part"] = len(i_in_new); new_sub["i_in"] = i_in_new
                        subs.append(new_sub)

                
                if len(i_in) > 200000:
                    # Temporary measure to get a catalogue in EdS-cond
                    if self.verbose: print('Reached size limit:', len(i_in), end = ' ', flush = True)
                    self.debug_out["exit"] = 'SZ'
                    break
                # Check a couple levels down if there are bound particles
                '''
                mean_vel = np.median(self.vel[list(i_in)], axis = 0)
                local_env = self.ngbs[i_cons][:] 
                
                current_level = np.array(local_env)
                next_level = []
                for level in range(5):
                    next_level = []
                    for part in current_level:
                        next_level.append(self.ngbs[part][:])
                    if len(next_level) == 0:
                        break
                    
                    
                    current_level = np.hstack(next_level)
                    local_env = np.hstack([local_env, current_level])
                     
                local_env = np.unique(local_env)
                v_in = self.vel[local_env] - mean_vel
                K = 0.5 * np.sum(v_in * v_in, axis = 1)
                E = K + self.phi_boost(local_env)       
                if np.all(E > self.phi_boost(i_cons)):
                    if self.verbose: print('Saddle point is not bound exitting...', len(i_in_new), end = ' ', flush = True)
                    self.debug_out["exit"] = 'US'
                    self.debug_out["mean_vel"] = mean_vel
                    self.debug_out["K"] = K#0.5*np.sum((self.vel[i_cons] - mean_vel)*(self.vel[i_cons] - mean_vel))
                    self.debug_out["E"] = E#self.phi_boost(i_cons)
                    self.debug_out["ind_end"] = i_cons
                    break
                '''
        ids = list(i_in)
        i_sad = ids[np.argmax(self.phi_boost(ids))]
        i_min = ids[np.argmin(self.phi_boost(ids))]
        n_part = len(i_in)
        res = {}
        res["i_min"] = i_min
        res["i_sad"] = i_sad
        res["n_part"] = n_part
        res["i_in"] = i_in
        if self.substruct:
            res["subs"] = subs         
        return res
    
    def is_bound(self, i_in, i_sad):
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
        i_in_arr = list(i_in)
        v_mean = np.median(self.vel[i_in_arr], axis = 0) # remove the mean velocity of the group
        v_in = self.vel[i_in_arr] - v_mean
        x_in = self.recentre_positions(self.pos[i_in_arr], self.x0)
        K = 0.5 * np.sum(v_in * v_in, axis = 1) + self._scale_factor * self.H_a(self._scale_factor) * np.sum(x_in * v_in, axis = 1)
        E = K + self._scale_factor**2 * self.phi_boost(i_in_arr) + 0.5*(self._Omega_m/2 + 1)*self._H0**2* self._scale_factor**-1 * np.sum(x_in * x_in, axis = 1)
        bound = E < self.phi_boost(i_sad)
        mask = np.zeros(len(i_in_arr),dtype = bool)
        mask[bound] = True
        return mask
        
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
        
        self.i_in = set()
        self.i_surf = set()
        # Verify that the minimu is not too far away.
        i0 = self.first_minimum(i0, r)
        # Find all particles with potential lower than minimum (This should only give back 1 particle)
        self.itt_check(i0)
        # Grow potential surface
        res = self.grow(self.i_in, self.i_surf)
        self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        #Binding check
        if self.no_binding: 
            return res
        mask = self.is_bound(res["i_in"], res["i_sad"])
        i_in_bound = np.array(list(res["i_in"]))[mask]
        res["i_in"] &= set(i_in_bound)
        if self.substruct:
            for i,sub in enumerate(res["subs"]):
                mask = self.is_bound(sub["i_in"], res["i_sad"])
                i_in_bound = np.array(list(sub["i_in"]))[mask]
                res["subs"][i]["i_in"] &= set(i_in_bound)
            
        if self.verbose:
            print(f"Unbound {res['n_part'] - len(res['i_in'])} particles", flush = True)
        res["n_part"] = len(res["i_in"])
        return res # output: i_min, i_sad, n_part(, subs, i_in)
    

        