# distutils: language = c++
import numpy as np
cimport numpy as cnp
from scipy.integrate import quad
from libcpp cimport bool as cbool
from math import floor
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC

cdef class ParticleAssigner:
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
    
    - fill_below: itterative watershed algorithm finding all conected particles with lower potentials segementing them into internal and surface sets which are saved internally.
    
    - fill_below_substructure: same as fill_below but explicitly takes the inner and surface sets as aguments instead of using internal variables.
    
    - sort_surface: sorts particles assigned to the surface set of the structure in order of increasing boosted potential. 
    
    - grow: grows the sturcture itteratively by including one by one the particles with the lowest bosted potential until a saddle point is found.
    
    - segment: main user function, selects all the connected particles that are assigned to the structure.
    
    '''
    
    cdef Py_ssize_t nparts
    cdef Py_ssize_t nngbs
    
    cdef long[:,:] ngbs
    cdef cnp.double_t[:] pot
    cdef cnp.double_t[:,:] pos
    cdef cnp.double_t[:,:] vel

    cdef cbool substruct
    cdef cbool verbose
    cdef cbool no_binding
    cdef str threshold

    cdef cnp.double_t Lbox
    cdef cnp.double_t _Omega_m
    cdef cnp.double_t _Omega_L
    cdef cnp.double_t _Omega_k
    cdef cnp.double_t _scale_factor
    cdef cnp.double_t _H0
    cdef cnp.double_t _delta_th

    cdef cnp.double_t[:] x0
    cdef cnp.double_t[:] acc0
    cdef cnp.uint8_t[:] visited

    cdef cnp.uint8_t[:] _computed
    cdef cnp.double_t[:] _phi 
    cdef cnp.int64_t[:] group
    cdef cnp.int64_t[:] subgroup
    cdef cnp.int64_t[:] parent

    cdef cnp.uint8_t[:] group_mask
    cdef cnp.uint8_t[:] surface_mask
    cdef long[:] surface_indices
    cdef long[:] surface_rank

    cdef cnp.uint8_t[:] subgroup_mask
    cdef cnp.uint8_t[:] subsurface_mask
    cdef long[:] subsurface_indices
    cdef long[:] subsurface_rank

    cdef cnp.uint8_t[:] bound_mask

    
    cdef cnp.int64_t _current_group
    cdef cnp.int64_t _current_subgroup
    
    cdef cnp.int64_t _current_group_size
    cdef cnp.int64_t _current_subgroup_size
    cdef cnp.int64_t _current_surface_size
    cdef cnp.int64_t _current_subsurface_size

    cdef cnp.double_t max_dist
    cdef cnp.double_t new_min
    cdef cnp.double_t _long_range_fac
        
    def __init__(self, cnp.ndarray[long, ndim = 2] ngbs, cnp.ndarray[double, ndim = 1] pot, cnp.ndarray[double, ndim = 2] pos, cnp.ndarray[double, ndim = 2] vel, 
                        double scale_factor = 1., double Omega_m = 1., double Lbox = 1000., str threshold = 'EdS-cond',
                 substruct = False, no_binding = False, verbose = False):
        
        self.nparts = ngbs.shape[0]
        self.nngbs = ngbs.shape[1]
        
        self.ngbs = ngbs
        self.pot = pot
        self.pos = pos
        self.vel = vel
        
        self.substruct = substruct
        self.verbose = verbose
        self.no_binding = no_binding
        self.threshold = threshold
        
        self.Lbox = Lbox
        self._Omega_m = Omega_m
        self._Omega_L = 1. - Omega_m
        self._Omega_k = 0.
        self._scale_factor = scale_factor
        self._H0 = 100. #In theory we should need to touch this
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
        self.surface_indices = np.zeros(pot.size, dtype = 'i8') - 1
        self.surface_rank = np.zeros(pot.size, dtype = 'i8') - 1
        
        self.subgroup_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_indices = np.zeros(pot.size, dtype = 'i8') - 1
        self.subsurface_rank = np.zeros(pot.size, dtype = 'i8') - 1
        
        self.bound_mask = np.zeros(pot.size, dtype = bool)
        
        #self._surface_size = 0
        self._current_group = 0
        self._current_subgroup = 0
        self._current_group_size = 0
        self._current_surface_size = 0
        self._current_subgroup_size = 0
        self._current_subsurface_size = 0
        #self._lowest_pot_id = -1
        
        self.new_min = 3.4e38 # this is just initialising the variable
        # In spherical symetry to get threshold at <delta> we need delta' = 2pi*<delta> - 1
        self._long_range_fac = 0.5 * (1 + (2*np.pi*self._delta_th - 1)) *self._Omega_m * self._scale_factor**2 * self._H0 * self._H0
        self.max_dist = 1. # We should only realy be worried once structures start getting biger than 1 Mpc 
        #self.i_in = set()
        #self.i_surf = set()
        #self.debug_out = {}
        
        return
    
    # ================== Cosmology ==============================
    # Everything should be in units of h
    
    def update_cosmology(self, scale_factor = None, Omega_m = None, Lbox = None):
        if Lbox != None:
            self.Lbox = Lbox
        if Omega_m != None:
            self._Omega_m = Omega_m
            self._Omega_L = 1. - Omega_m
            self._Omega_k = 0.
        if scale_factor != None:
            self._scale_factor = scale_factor
        self.set_delta_th(self.threshold)
        return 
    
    cdef double H_a(self, a):
        return self._H0 * np.sqrt(self._Omega_m/a*a*a+self._Omega_k/a*a+self._Omega_L)
        
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
                print(xi, end= " ")
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
    
    def to_bool_array(self, lst):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] res
        res = np.array(lst, dtype=bool)
        return res
    
    cpdef long_select(self, arr, cond):
        cdef cnp.ndarray[long] indices
        cdef cnp.ndarray[long, cast = True] res
        cdef list lst
        cdef long i,j
        lst = []
        indices = np.where(self.to_bool_array(cond))[0]
        res = np.zeros(len(indices), dtype = 'i8')
        for i,j in enumerate(indices):
            res[i] = arr[j]
        return res
    
    cpdef double_select(self, arr, cond):
        cdef cnp.ndarray[long] indices
        cdef cnp.ndarray[double, cast = True] res
        cdef list lst
        cdef long i,j
        lst = []
        indices = np.where(self.to_bool_array(cond))[0]
        res = np.zeros(len(indices), dtype = 'f8')
        for i,j in enumerate(indices):
            res[i] = arr[j]
        return res
    
    cpdef double_select_2D(self, arr, cond):
        cdef cnp.ndarray[long] indices
        cdef cnp.ndarray[double, ndim = 2, cast = True] res
        cdef list lst
        cdef long i,j
        lst = []
        indices = np.where(self.to_bool_array(cond))[0]
        res = np.zeros(np.shape(arr), dtype = 'f8')
        for i,j in enumerate(indices):
            res[i,:] = np.asarray(arr[j])
        return res
        
    cpdef void set_current_group(self, long i):
        self._current_group = i
        return
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_group_ledger(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.group)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_subgroup_ledger(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.subgroup)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_surface_indices(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.surface_indices)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_surface_ranks(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.surface_rank)
        return res
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_surface_mask(self):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.surface_mask)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_group_particles(self, i):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.where(np.asarray(self.group) == i)[0]
        return res
    
    cpdef list get_subgroups(self, i):
        cdef list res
        cdef cnp.ndarray[long, ndim = 1, cast = True] ids = self.get_group_particles(i)
        cdef cnp.ndarray[long, ndim = 1, cast = True] subgroups = self.get_subgroup_ledger()[ids]
        cdef long index, j
        res = [ids[subgroups == j] for j in range(0,int(np.max(subgroups))+1)]
        return res

    
    
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
    
    def recentre_positions_numpy(self,pos, x0):
        '''
        Function which recentres and wraps the position vector 'pos' to a new centre at 'x0'.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.
        pos: (array of d-floats) positions of of particles in d-dimensional space.

        '''
        res = (pos - x0 - self.Lbox/2)%self.Lbox - self.Lbox/2
        return res
    
    cdef cnp.ndarray[cnp.double_t, ndim = 1] recentre_positions(self, cnp.double_t[:] pos, cnp.double_t[:] x0):
        '''
        Function which recentres and wraps the position vector 'pos' to a new centre at 'x0'.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.
        pos: (array of d-floats) positions of of particles in d-dimensional space.

        '''
        cdef cnp.ndarray[cnp.double_t, ndim = 1] res = np.zeros(np.shape(pos))
        cdef int i = 0
        for i in range(len(pos)):
            res[i] = (pos[i] - x0[i] - self.Lbox/2)%self.Lbox - self.Lbox/2
        return res
    
    cdef cnp.double_t phi_boost(self, long i):
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
            
        cdef cnp.ndarray[double, ndim = 1, cast = True] x
        cdef double temp_xx, temp_xa
        cdef int k
        cdef double res
        
        if self._computed[i]:
            res = self._phi[i]
        else:
            x = self.recentre_positions(self.pos[i],self.x0)
            self._computed[i] = True
            temp_xx = 0.0
            temp_xa = 0.0
            for k in range(len(x)):
                temp_xx += x[k]*x[k]
                temp_xa += x[k]*self.acc0[k]
            self._phi[i] = self.pot[i] + temp_xa - self._long_range_fac * temp_xx
            res = self._phi[i]
        return res
    
    def get_phi_boost(self, indices):
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
            
        #cdef cnp.ndarray[double, ndim = 1, cast = True] x
        #cdef double temp_xx, temp_xa
        #cdef int k
        #cdef double res
        res = np.zeros(indices.size, dtype = 'f8')
        cond = self.to_bool_array(self._computed)[indices]
        #if self._computed[i]:
        #    res = self._phi[i]
        res[cond] = np.asarray(self._phi)[indices[cond]]
        x = np.zeros(len(self.pos[0]))
        for j,i in enumerate(indices[np.logical_not(cond)]):
            x = self.recentre_positions(self.pos[i],self.x0)
            #self._computed[i] = True
            temp_xx = 0.0
            temp_xa = 0.0
            for k in range(len(x)):
                temp_xx += x[k]*x[k]
                temp_xa += x[k]*self.acc0[k]
            res[j] = self.pot[i] + temp_xa - self._long_range_fac * temp_xx
            #res = self._phi[i]
        return res
    #####=================== Neighbours ==================
    
    def subfind_neighbours(self):
        '''
        Function altering the list of neighbours to obtain a connectivity equivalent to that used within the subfind algorithm. See Springel et al. 2001, MNRAS 328, 726–750 for details on the subfind algorithm. Warning this functgion will rewrite the entire neighbours list and as such can take an extremely long amount of time if used on too many particles.
        
        Parameters:
        ----------
        None
        '''
        
        recip_ngbs = list(self.ngbs)
        
        for i in range(self.pos.shape[0]):
            cond = self.phi_boost(recip_ngbs[i]) < self.phi_boost(i)
            recip_ngbs[i] = np.array(recip_ngbs[i][cond][:2])
    
        for i in range(self.pos.shape[0]):
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
        
        for i in range(self.pos.shape[0]):
            for j in recip_ngbs[i]:
                if i not in set(recip_ngbs[j]):
                    recip_ngbs[j] = np.array(list(recip_ngbs[j]) + [i,])
        self.ngbs = recip_ngbs
        return
    
    #####=================== Sorting ==================
    
    cpdef long secant_search(self, cnp.double_t elem, cnp.double_t[:] l):
        cdef int MAXIT = 30
        cdef int i = 0
        cdef long N = len(l) - 1
        cdef long x1 = N
        cdef long rts = 0
        cdef long x2 = rts
        cdef cnp.double_t f1 = l[x1] - elem
        cdef cnp.double_t f = l[x2] - elem
        cdef cnp.double_t f_temp = f
        cdef cnp.double_t dx = 0
        
        if elem > l[-1]:
            return N+1
        if elem < l[0]:
            return 0

        if np.abs(f1) < np.abs(f):
            rts = x1
            x1 = x2
            f_temp = f; f = f1; f1 = f_temp
        else:
            x1 = x1
            rts = x2

        for i in range(MAXIT):
            
            dx = (x1 - rts) * f/(f-f1)
            x1 = rts
            f1 = f
            rts += floor(dx)
        
            if rts < 0: rts = 0
            if rts > N: rts = N
            f = l[rts] - elem
            if np.abs(dx) < 1 or f == 0 or (x1 - rts) == 0: return rts+1
            elif np.abs(floor(dx)) == 1: return x1 + 1
        raise ValueError("Root Finder failed to converge")
        return rts
    
    cpdef long FP_search(self, cnp.double_t elem, cnp.double_t[:] l):
        cdef int MAXIT = 30
        cdef int i = 0
        cdef long N = len(l) - 1
        cdef long xl = 0
        cdef long x2 = N
        cdef long xh = x2
        cdef long dx = np.abs(xl - x2)
        cdef long DEL = 0
        cdef cnp.double_t fl = l[xl] - elem
        cdef cnp.double_t fh = l[xh] - elem
        cdef cnp.double_t f = 0.0
        cdef long rtf
        
        
        if N == 0:
            if elem < l[0]: return 0
            else: return 1
        if elem > l[-1]:
            return N+1
        if elem < l[0]:
            return 0

        dx = np.abs(xh - xl)
        for i in range(MAXIT):
            rtf = xl + floor(dx * fl/(fl - fh)) 
            f =  l[rtf] - elem 
            if f < 0.0:
                DEL = xl - rtf
                xl = rtf
                fl = f
            else:
                DEL = xh - rtf
                xh = rtf
                fh = f
            dx = xh - xl
            if np.abs(DEL) < 1 or f == 0.0 or (fl - fh) == 0.0 : return rtf + 1 
            elif np.abs(DEL) == 1: return xl + 1
        raise ValueError("Root Finder failed to converge")
        return rtf + 1
    
    cpdef long binary_search(self, cnp.double_t elem, cnp.double_t[:] l):
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
        #if not self.is_sorted(l): raise ValueError("The list provided is not sorted in increasing order.")
        
        cdef int JMAX = 100
        cdef int j
        
        cdef long N = len(l) - 1
        cdef long dx = N
        cdef long xmid = (N+1)//2
        cdef long rtb = 0
        cdef long rtt = N
        cdef cnp.double_t f = l[0] - elem
        cdef cnp.double_t fmid = l[N] - elem
        if elem > l[-1]:
            return N+1
        if elem < l[0]:
            return 0
        for j in range(JMAX):
            #print(fmid, xmid, dx)
            dx = (rtt - rtb)//2
            xmid = rtb + dx

            fmid = l[xmid] - elem
            if fmid <= 0.: rtb = xmid
            else: rtt = xmid
            if abs(dx) < 1 or fmid == 0.0: 
                return rtb + 1

        raise ValueError("Root Finder failed to converge")
        return rtb + 1
   
    cdef cnp.ndarray[long] unmix(self, cnp.ndarray[long] a, cnp.ndarray[long] order):
        cdef cnp.ndarray[long] a_new = np.zeros(len(a), dtype = long)
        for i,j in enumerate(order):
            a_new[j] = a[i]
        return a_new

    cpdef cbool is_sorted(self, cnp.double_t[:] a):
        '''
        Function checking if array a is sorted in ascending order
        
        Parameters:
        ----------
        a: (array of float) array to check
        
        Returns:
        ----------
        res: (bool) True is the array is ascending
        '''
        cdef long k
        for k in range(len(a)-1):
            if a[k+1] < a[k]:
                return False
        return True #np.all(np.diff(a) >= 0)
    
    cdef sort_surface(self, cnp.uint8_t[:] surface_mask):
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
        
        cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.ndarray[cnp.double_t, ndim = 1, cast = True] phi_surf 
        cdef cnp.ndarray[long, ndim = 1, cast = True] args
        cdef cnp.ndarray[long, ndim = 1, cast = True] indices
        cdef long ind, arg, j
        id_surf = np.where(surface_mask)[0]
        phi_surf = np.zeros(len(id_surf))
        for j in range(len(id_surf)):
            phi_surf[j] = self.phi_boost(id_surf[j])
        args = np.argsort(phi_surf)
        indices = np.zeros(args.size, dtype = 'i8')
        for j, arg in enumerate(args):
            indices[j] = id_surf[arg] 
        return id_surf, indices

    
    cdef cnp.int64_t insert_mask_potential_sorted(self, long i, long[:] indices, long[:] ranks, cnp.uint8_t[:] mask, cnp.int64_t size):
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

        cdef long rank, j
        
        if size == 0:
            indices[0] = i
            ranks[i] = 0
            mask[i] = True
            size +=1
            return size
        
        cdef cnp.double_t[:] phi_arr
        
        phi_arr = np.zeros(size)
        for j in range(size):
            phi_arr[j] = self.phi_boost(indices[j])
        
        #if not self.is_sorted(phi_arr): phi_arr = np.sort(phi_arr) # This is a temporary fix... which slows the code a lot. Have to find where the ordering is lost
        rank = self.binary_search(self.phi_boost(i), phi_arr)
        #rank = self.secant_search(self.phi_boost(i), phi_arr)
        #rank = self.FP_search(self.phi_boost(i), phi_arr)
        for j in reversed(range(rank, size)):
            indices[j+1] = indices[j]
            ranks[indices[j+1]] += 1 

        size += 1
        indices[rank] = i
        ranks[i] = rank
        mask[i] = True
        return size
    
    cdef cnp.int64_t remove_mask_potential_sorted(self, long i,  long[:] indices, long[:] ranks, cnp.uint8_t[:] mask, cnp.int64_t size):
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
        cdef long rank, index, j 
        
        rank = ranks[i]
        for j in range(rank, size):
            index = indices[j]
            ranks[index] -= 1
            indices[j] = indices[j+1]
        size -= 1
        ranks[i] = -1
        mask[i] = False
        return size
    
    # ================== Particle Assignment ====================
    
    cpdef long first_minimum(self, long i0, double r = 1):
        cdef int counter = 1
        cdef long i = i0
        cdef long j
        cdef double dist = 0.0
        cdef cnp.ndarray[double, ndim = 1, cast = True] x
        cdef cnp.ndarray[double, ndim = 1, cast = True] phi_ngbs
        cdef long[:] ngbs_temp


        ngbs_temp = self.ngbs[i]
        phi_ngbs = np.zeros(len(ngbs_temp))
        for j in range(len(ngbs_temp)):
            phi_ngbs[j] = self.phi_boost(ngbs_temp[j])
        while np.min(phi_ngbs) < self.phi_boost(i):
            counter += 1
            i = ngbs_temp[np.argmin(phi_ngbs)]
            ngbs_temp = self.ngbs[i]
            phi_ngbs = np.zeros(len(ngbs_temp))
            for j in range(len(ngbs_temp)):
                phi_ngbs[j] = self.phi_boost(ngbs_temp[j])
                
            if counter % 100 == 0:
                x = self.recentre_positions(self.pos[i], self.x0)
                dist = np.sqrt(np.sum(x*x))
                if dist > r:
                    raise RecursionError(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                if counter > 1000:
                    raise RecursionError("The minimum was not found after 1000 steps.\nThis may indicate that there is either no miminum or that you are starting index is too far away.")
        x = self.recentre_positions(self.pos[i], self.x0)
        dist = np.sqrt(np.sum(x*x))
        if dist > r:
            raise RecursionError(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                  \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
        return i
    
    cpdef void fill_below(self, cnp.int64_t i):
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
        cdef double phi_curr = self.phi_boost(i)
        #cdef cnp.ndarray[long, ndim = 1, cast = True] ngbs_temp
        cdef long[:] ngbs_temp
        cdef long j, k, l, index
        cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.ndarray[long, ndim = 1, cast = True] indices
        
        ngbs_temp = self.ngbs[i]
        for k in range(len(ngbs_temp)):
            if not self.visited[ngbs_temp[k]]:
                self.surface_mask[ngbs_temp[k]] = True
        # Sort the surface
        id_surf, indices = self.sort_surface(self.surface_mask) # This function outputs a numpy array
        self._current_surface_size = len(indices)
        for k, index in enumerate(indices):
            self.surface_indices[k] = index
            self.surface_rank[index] = k
           
        # Take surface particles with a potential lower that current
        j = self.surface_indices[0]
        
        while self.phi_boost(j) - phi_curr < 0.0:
            
            self.group[j] = self._current_group
            self.group_mask[j] = True
            self.parent[j] = self._current_group
            self._current_group_size +=1
            self.visited[j] = True
            
            self._current_surface_size = self.remove_mask_potential_sorted(j, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
            ngbs_temp = self.ngbs[j]
            
            for k in ngbs_temp:
                if self.surface_mask[k] or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    self._current_surface_size = self.insert_mask_potential_sorted(k, self.surface_indices,self.surface_rank, self.surface_mask, self._current_surface_size)
            j = self.surface_indices[0]
        return
    

    
    cdef void fill_below_substructure(self, long i, double phi_min):
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
        cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        #cdef cnp.ndarray[long, ndim = 1, cast = True] order
        cdef cnp.ndarray[cnp.double_t, ndim = 1, cast = True] x
        cdef double dist
        cdef double phi_curr = self.phi_boost(i)
        cdef long j, k, index
        cdef long[:] ngbs_temp
        cdef int counter = 1
        cdef double phi_j
        
        # Start by adding the refference particle to the group
        self.subgroup_mask[i] = True
        self.visited[i] = True
        # Calculate it's potential and define its neighbours as surface particles
        
        ngbs_temp = self.ngbs[i]
        for k in range(len(ngbs_temp)):
            if not self.visited[ngbs_temp[k]]:
                self.subsurface_mask[ngbs_temp[k]] = True
        # Sort the surface
        id_surf, indices = self.sort_surface(self.subsurface_mask) # This function outputs a numpy array
        self._current_subsurface_size = len(indices)
        for k, index in enumerate(indices):
            self.subsurface_indices[k] = index
            self.subsurface_rank[index] = k
        
        # Take surface particles with a potential lower that current
    
        j = self.subsurface_indices[0]
        
        while self.phi_boost(j) - phi_curr < 0:
            counter += 1        
            # By taking this one we ensure we are always going down the lowest brach first.
            phi_j = self.phi_boost(j)
            
            if counter % 100 == 0: # Check if we haven't wandered too far away
                x = self.recentre_positions(self.pos[j], self.x0)
                dist = np.sqrt(np.sum(x*x))
                if dist > 3*self.max_dist:
                    raise RecursionError(f"The new minimum is {float(dist):.3} (> {float(3*self.max_dist):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                if self._current_subgroup_size > 4*self._current_group_size:
                    raise RecursionError(f"Trying to add in a structure which is much larger than the current structure. Exiting.")
                    
            # If we are in this we already know that j has a lower potential
            # So we can add it directly and remove it from the surface
            self._current_subsurface_size = self.remove_mask_potential_sorted(j, self.subsurface_indices,self.subsurface_rank, self.subsurface_mask, self._current_subsurface_size)
            if self.visited[j]:
                continue
            self.subgroup_mask[j] = True
            self.visited[j] = True
            self._current_subgroup_size += 1 
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
                    self._current_subsurface_size = self.insert_mask_potential_sorted(k, self.subsurface_indices,self.subsurface_rank, self.subsurface_mask, self._current_subsurface_size)
            # Take surface particles with a potential lower that current
            
            j = self.subsurface_indices[0]
        return
    
    cpdef tuple grow(self):
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
        cdef double phi_min, phi_max
        
        cdef long i_prev, n_part
        cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.ndarray[long, ndim = 1, cast = True] indices
        #cdef cnp.ndarray[long, ndim = 1, cast = True] surf_temp
        cdef long j, k, index, elem
        
        cdef long i_cons
        cdef cnp.double_t[:] phi_boost_ngbs
        cdef cnp.ndarray[long, ndim = 1, cast = True] group_particles
        cdef long[:] ngbs_temp
        #cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] low_pot_cond
        cdef cbool low_pot_cond
        #cdef cnp.ndarray[long, ndim = 1, cast = True] i_low
        cdef cbool if_cond, size_cond
        cdef timespec ts
        cdef double start_loop, end_loop, start, end
        
        cdef cnp.ndarray[cnp.double_t, ndim = 1, cast = True] x
        cdef double dist
        
        group_particles = self.get_group_particles(self._current_group)
        n_part = group_particles.size
        phi_min = self.phi_boost(group_particles[0])
        for k in group_particles:
            x = self.recentre_positions(self.pos[k], self.x0)
            dist = np.sqrt(np.sum(x*x))
            if dist > self.max_dist:
                self.max_dist = dist
            if self.phi_boost(k) < phi_min:
                phi_min = self.phi_boost(k)
        
        i_prev = -1
        
        id_surf, indices = self.sort_surface(self.to_bool_array(self.surface_mask)) # This function outputs a numpy array
        self._current_surface_size = len(indices)
        for k, index in enumerate(indices):
            self.surface_indices[k] = index
            self.surface_rank[index] = k

        for _ in range(len(self.pot)): # <== I just don't want to write 'while True:'
            
            if not np.any(self.surface_mask):
                if self.verbose: print('Surface empty', flush = True)
                break
            
            i_cons = self.surface_indices[0]
            
            if i_cons == i_prev:
                raise RecursionError(f"The algorithm seems to have gotten stuck at i = {i_cons} with {self._current_group_size} particles assigned and {self._current_surface_size} pending...")
            i_prev = i_cons
            self.visited[i_cons] = True
            ngbs_temp = self.ngbs[i_cons]
            
            low_pot_cond = False
            for j in range(len(ngbs_temp)):
                if self.phi_boost(ngbs_temp[j]) < self.phi_boost(i_cons):
                    low_pot_cond = True
                    break

            x = self.recentre_positions(self.pos[i_cons], self.x0)
            dist = np.sqrt(np.sum(x*x))
            if dist > self.max_dist:
                self.max_dist = dist
            
            if not low_pot_cond:
                #This can happen when hitting the boundaries
                #self.surface_indices, self.surface_rank, self.surface_mask = 
                self._current_surface_size = self.remove_mask_potential_sorted(i_cons, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
                continue
            
            if_cond = True
            
            for k, elem in enumerate(ngbs_temp):
                if self.phi_boost(elem) < self.phi_boost(i_cons):
                    if_cond *= self.group_mask[elem]
            
            if if_cond:
                # Tag i_cons as part of the main group
                self.group[i_cons] = self._current_group
                self.group_mask[i_cons] = True
                self.parent[i_cons] = self._current_group
                self._current_group_size += 1 
                self._current_surface_size = self.remove_mask_potential_sorted(i_cons, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
                for k in self.ngbs[i_cons]:
                    if self.surface_mask[k] or self.visited[k]:
                        # Avoid duplicates or going back to a particle that has already been visited
                        continue
                    else: 
                        # Insert particle in sorted order
                        self._current_surface_size = self.insert_mask_potential_sorted(k, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
                

            else:
                self.visited[i_cons] = False
                self.new_min = self.phi_boost(i_cons)
                
                self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)
                self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
                self.subsurface_indices = np.zeros(self.pot.size, dtype = 'i8') - 1
                self.subsurface_rank = np.zeros(self.pot.size, dtype = 'i8') - 1
                self._current_subsurface_size = 0
                self._current_subgroup_size = 0
                try:
                    self.fill_below_substructure(i_cons, phi_min)
                except RecursionError:
                    if self.verbose: print('Traveled too far exitting... ', self._current_subgroup_size, end = ' ', flush = True)
                    
                    self.visited[np.logical_or(self.subgroup_mask,self.subsurface_mask)] = False
                    break
                
                if self.new_min <= phi_min:
                    # Moved into a lower potential well => exit
                    if self.verbose: print('Found lower minimum:', self._current_subgroup_size, end = ' ', flush = True)
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
                            self._current_surface_size = self.insert_mask_potential_sorted(k, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
                    
                    size_cond = self._current_subgroup_size > 20
                    if size_cond:
                        self._current_subgroup += 1
                    for k in np.where(self.subgroup_mask)[0]:
                        self.group[k] = self._current_group
                        self.group_mask[k] = True
                        self.parent[k] = self._current_group
                        self._current_group_size += 1
                        if size_cond:
                            self.subgroup[k] = self._current_subgroup
                        x = self.recentre_positions(self.pos[k], self.x0)
                        dist = np.sqrt(np.sum(x*x))
                        if dist > self.max_dist:
                            self.max_dist = dist
                            
                    if self.surface_mask[i_cons]:
                        #self.surface_indices, self.surface_rank, self.surface_mask = 
                        self._current_surface_size = self.remove_mask_potential_sorted(i_cons, self.surface_indices, self.surface_rank, self.surface_mask, self._current_surface_size)
                    
                if self._current_group_size > 0.25*self.pot.size:
                    # Temporary measure to get a catalogue in EdS-cond
                    if self.verbose: print('Reached size limit:', self._current_group_size, end = ' ', flush = True)
                    
                    break
        
        cdef cnp.ndarray[long, ndim = 1, cast = True] ids 
        ids = self.get_group_particles(self._current_group)
        cdef long i_min, i_sad, i
        phi_min = self.phi_boost(ids[0])
        phi_max = self.phi_boost(ids[0])
        
        for i in range(len(ids)):
            phi_curr = self.phi_boost(ids[i])
            if phi_curr > phi_max:
                phi_max = phi_curr
                i_sad = ids[i]
            if phi_curr < phi_min:
                phi_min = phi_curr
                i_min = ids[i]
        
        return i_min, i_sad
    
    cpdef is_bound(self, long i_min, long i_sad):
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
        cdef cnp.ndarray[long, ndim = 1, cast = True] i_in_arr
        
        cdef cnp.ndarray[double, ndim = 1, cast = True] v_mean 
        cdef cnp.ndarray[double, ndim = 2, cast = True] v_in
        cdef cnp.ndarray[double, ndim = 1, cast = True] v
        cdef cnp.ndarray[double, ndim = 1, cast =True] x
        cdef cnp.ndarray[double, ndim = 1, cast = True] phi_p
        cdef cnp.ndarray[double, ndim = 1, cast = True] K
        cdef cnp.ndarray[double, ndim = 1, cast = True] E
        cdef double phi_p_sad, temp_xx, temp_xv, temp_vv
        cdef long i = 0
        cdef long index
        cdef int j = 0
        
        i_in_arr = np.where(self.group_mask)[0]
        v_in = np.zeros((len(i_in_arr), self.vel.shape[1]))
        K = np.zeros(len(i_in_arr))
        phi_p = np.zeros(len(i_in_arr))
        E = np.zeros(len(i_in_arr)) 
        
        for j,i in enumerate(i_in_arr):
            v_in[j,:] = self.vel[i]
        v_mean = np.median(v_in, axis = 0) # remove the mean velocity of the group
        #v_in -= v_mean
        
        
        x = self.recentre_positions(self.pos[i_sad],self.pos[i_min])
        #v = v_in[i_sad] - v_mean
        temp_xx = 0.0
        #temp_xv = 0.0
        #temp_vv = 0.0
        for j in range(len(x)):
            temp_xx += x[j]*x[j]
            #temp_vx += v[j]*x[j]
            #temp_vv += v[j]*v[j]
        phi_p_sad = self._scale_factor**2 * self.phi_boost(i_sad) + 0.5*(self._Omega_m/2 + 1)*self._H0**2* self._scale_factor**-1 * temp_xx
        
        for i, index in enumerate(i_in_arr):
            x = self.recentre_positions(self.pos[index],self.pos[i_min])
            v = v_in[i,:] - v_mean
            temp_xx = 0.0
            temp_xv = 0.0
            temp_vv = 0.0
            for j in range(len(x)):
                temp_xx += x[j]*x[j]
                temp_xv += v[j]*x[j]
                temp_vv += v[j]*v[j]
            #x2 = np.sum(x*x, axis = 1)
            
            K[i] = 0.5 * temp_vv + self._scale_factor * self.H_a(self._scale_factor) * temp_xv
            phi_p[i] = self._scale_factor**2 * self.phi_boost(i) + 0.5*(self._Omega_m/2 + 1)*self._H0**2* self._scale_factor**-1 * temp_xx
            E[i] = K[i] + phi_p[i] # <= Converted to physical potential
            if E[i] < phi_p_sad:
                self.bound_mask[index] = True
        
        
        #bound = E < phi_p_sad
        #for i, index in i_in_arr:
        #    self.bound_mask[i] = True
        return self.bound_mask

    
    cpdef segment(self, long i0, cnp.double_t[:] acc0, r = 1):
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
        cdef long i_min, i_sad
        self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        self.acc0 = acc0
        self.x0 = self.pos[i0]
        
        self._current_group += 1
        self._current_subgroup = 0
        # Verify that the minimum is not too far away.
        i0 = self.first_minimum(i0, r)
        # Find all particles with potential lower than minimum (This should only give back 1 particle)
        self.fill_below(i0)
        # Grow potential surface
        i_min, i_sad = self.grow()
        self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        #Binding check
        if self.no_binding: 
            return np.where(self.group_mask)[0], i_min, i_sad
        self.bound_mask = self.is_bound(i_min, i_sad)
        return np.where(np.logical_and(self.group_mask,self.bound_mask))[0], i_min, i_sad # output: i_min, i_sad, n_part(, subs, i_in)
    
    cpdef void reset(self):
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
        self.surface_rank = np.zeros(self.pot.size, dtype = 'i8') - 1
        self.surface_indices = np.zeros(self.pot.size, dtype = 'i8') - 1
        
        self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_rank = np.zeros(self.pot.size, dtype = 'i8') - 1
        self.subsurface_indices = np.zeros(self.pot.size, dtype = 'i8') - 1
        
        self.bound_mask = np.zeros(self.pot.size, dtype = bool)
        
        self._current_group = 0
        self._current_subgroup = 0
        self._current_group_size = 0
        self._current_subgroup_size = 0
        self._current_surface_size = 0
        self._current_subsurface_size = 0
        
        self.new_min = 3.4e38
        return 