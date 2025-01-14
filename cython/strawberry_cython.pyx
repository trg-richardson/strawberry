# distutils: language = c++
import numpy as np
cimport numpy as cnp
from scipy.integrate import quad
from libcpp cimport bool as cbool
from libcpp.pair cimport pair as cpair
#from math import floor
#from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC

cdef extern from "cpp_priority_queue.hpp":
    cdef cppclass cpp_pq:
        cpp_pq(...) except +
        void push(cpair[cnp.double_t,long])
        cpair[cnp.double_t,long] top()
        void pop()
        cbool empty()
        
cdef cbool compare_first(cpair[cnp.double_t, long] a, cpair[cnp.double_t, long] b):
    return a.first > b.first #flipping this sign says in which order we want the queue "< largest first" , "> smallest first"

cpdef void test_queue(cnp.ndarray[long, ndim = 1] indices, cnp.ndarray[cnp.double_t, ndim = 1] pots):
    cdef cpp_pq queue = cpp_pq(compare_first)
    cdef cpp_pq copy_queue = cpp_pq(compare_first)
    cdef cpair[cnp.double_t, long] elem
    cdef int i
    
    for i in range(len(indices)):
        elem = (pots[i], indices[i])
        queue.push(elem)
    copy_queue = queue
    while not queue.empty():
        elem = queue.top() 
        queue.pop()
        print(elem)
    if copy_queue.empty():
        print("The copy is shallow")
    else:
        print("The copy is deep")
    return 

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
    cdef long[:] ids_fof
    
    cdef cbool verbose
    cdef cbool no_binding
    cdef str threshold

    cdef cnp.double_t Lbox
    cdef cnp.double_t _Omega_m
    cdef cnp.double_t _Omega_L
    cdef cnp.double_t _Omega_k
    cdef cnp.double_t _scale_factor
    #cdef cnp.double_t _scale_factor3
    cdef cnp.double_t _H0
    cdef cnp.double_t _Ha
    cdef cnp.double_t _Hd
    cdef cnp.double_t _delta_th

    cdef long i0
    cdef cnp.double_t[:] x0
    cdef cnp.double_t[:] acc0
    cdef cnp.uint8_t[:] visited

    cdef cnp.uint8_t[:] _computed
    cdef cpp_pq _computed_queue
    cdef cnp.double_t[:] _phi 
    cdef cnp.int64_t[:] group
    cdef cnp.int64_t[:] subgroup
    #cdef cnp.int64_t[:] parent

    cdef cnp.uint8_t[:] group_mask
    cdef cnp.uint8_t[:] surface_mask
    cdef cpp_pq surface_queue
    cdef cpp_pq group_queue
    #cdef long[:] surface_indices
    #cdef long[:] surface_rank

    cdef cnp.uint8_t[:] subgroup_mask
    cdef cnp.uint8_t[:] subsurface_mask
    cdef cpp_pq subsurface_queue
    cdef cpp_pq subgroup_queue
    #cdef long[:] subsurface_indices
    #cdef long[:] subsurface_rank

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
    cdef cnp.double_t _zeta
    
    cdef cbool _too_far
        
    def __init__(self, cnp.ndarray[long, ndim = 2] ngbs, cnp.ndarray[double, ndim = 1] pot, cnp.ndarray[double, ndim = 2] pos, cnp.ndarray[double, ndim = 2] vel, 
                 cnp.double_t scale_factor = 1., cnp.double_t Omega_m = 1., cnp.double_t Lbox = 1000., cnp.double_t H0 = 100, str threshold = 'EdS-cond',
                 no_binding = False, verbose = False, cnp.ndarray[long, ndim = 1] ids_fof = np.empty([0], dtype = 'i8'), cbool custom_delta = False,  cnp.double_t delta = 0.):
        
        self.nparts = ngbs.shape[0]
        self.nngbs = ngbs.shape[1]
        
        self.ngbs = ngbs
        self.pot = pot
        self.pos = pos
        self.vel = vel
        self.ids_fof = ids_fof
        
        self.verbose = verbose
        self.no_binding = no_binding
        self.threshold = threshold
        
        self.Lbox = Lbox
        self._Omega_m = Omega_m
        self._Omega_L = 1. - Omega_m
        self._Omega_k = 0.
        self._scale_factor = scale_factor
        #self._scale_factor3 = scale_factor**(1/3.)
        self._H0 = H0
        self._Ha = self.H_a(self._scale_factor)
        self._Hd = self.H_dot(self._scale_factor)
        if custom_delta:
            self._delta_th = delta
        else:
            self._delta_th = self.get_delta_th(threshold)
        self.i0 = -1
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(pot.size, dtype = bool) # We may want to review these assignements
        
        self._computed = np.zeros(pot.size, dtype = bool)
        self._computed_queue = cpp_pq(compare_first)
        self._phi = np.zeros(pot.size, dtype = 'f8') 
        self.group = np.zeros(pot.size, dtype = 'i8')
        self.subgroup = np.zeros(pot.size, dtype = 'i8')
        
        self.group_mask = np.zeros(pot.size, dtype = bool)
        self.surface_mask = np.zeros(pot.size, dtype = bool)
        self.surface_queue = cpp_pq(compare_first)
        self.group_queue = cpp_pq(compare_first)
        
        self.subgroup_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(pot.size, dtype = bool)
        self.subsurface_queue = cpp_pq(compare_first)
        self.subgroup_queue = cpp_pq(compare_first)
        
        self.bound_mask = np.zeros(pot.size, dtype = bool)
        
        self._current_group = 0
        self._current_subgroup = 0
        self._current_group_size = 0
        self._current_surface_size = 0
        self._current_subgroup_size = 0
        self._current_subsurface_size = 0

        
        self.new_min = 3.4e38 # this is just initialising the variable
        self._too_far = False
        # In spherical symetry to get threshold at <delta> we need delta' = 2pi*<delta> - 1
        self._long_range_fac =  0.25 * np.pi * self._delta_th *self._Omega_m * self._scale_factor**-3 * self._H0 * self._H0
        if verbose:
            print(f"long range factor: {self._long_range_fac}")
        self.max_dist = 1. # We should only realy be worried once structures start getting biger than 1 Mpc 
        
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
            #self._scale_factor3 = scale_factor**(1/3.)
            self._Ha = self.H_a(scale_factor)
            self._Hd = self.H_dot(scale_factor)
        self._delta_th = self.get_delta_th(self.threshold)
        self._long_range_fac = 0.25 * np.pi * self._delta_th * self._Omega_m * self._scale_factor**-3 * self._H0 * self._H0
        self.reset_computed_particles()
        return 
    
    def H_a(self, a):
        return self._H0 * np.sqrt(self._Omega_m/a**(3)+self._Omega_k/a**(2)+self._Omega_L)

    def H_dot(self, a):
        return -self._H0 / 2 / np.sqrt(self._Omega_m/a**(3)+self._Omega_k/a**(2)+self._Omega_L) * (3*self._Omega_m/a**(4) + 2*self._Omega_k/a**(3))
        
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
    
    def get_delta_th(self, threshold):
        if 'EdS' in threshold:
            if "cond" in threshold:
                res = 0.
            elif "coll" in threshold:
                res = 3/5*(3*np.pi/2)**(2/3.)
            elif "ta-lin" in threshold:
                res = 3/5*(3*np.pi/4)**(2/3.)
            elif "ta-eul" in threshold:
                res = 9*np.pi**2/16 - 1
            else:
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
        elif 'LCDM' in threshold:
            t_c = self.time_of_a(self._scale_factor)
            t_ta = t_c/2
            a_ta = self.Newton_Raphson(f = lambda a: self.time_of_a(a) - t_ta, xi = 0.1, dx = 1e-6, tol = 1e-6)
            zeta = self.get_zeta(a_ta) 
            if "cond" in threshold:
                res = 9/10 * (2 * self.w(self._scale_factor))**(1/3)
            elif "coll" in threshold:
                res = 3/5 * self.g(self._scale_factor) * (1 + zeta) * (self.w(self._scale_factor)/zeta)**(1./3.)
            elif "ta-lin" in threshold:
                res = 3/5 * self.g(a_ta) * (1 + zeta) * (self.w(a_ta)/zeta)**(1./3.)
            elif "ta-eul" in threshold:
                zeta = self.get_zeta(self._scale_factor) 
                res = self._Omega_L/self._Omega_m * self._scale_factor**3/zeta - 1
            else:
                raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
        else:
            raise ValueError(f'threshold definition {threshold} was not recognized options include:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
        return res
    
    

    
    # ================== Utility Methods ========================
    
    def to_bool_array(self, lst):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] res
        res = np.array(lst, dtype=bool)
        return res
    
    #cpdef long_select(self, arr, cond):
    #    cdef cnp.ndarray[long] indices
    #    cdef cnp.ndarray[long, cast = True] res
    #    cdef list lst
    #    cdef long i,j
    #    lst = []
    #    indices = np.where(self.to_bool_array(cond))[0]
    #    res = np.zeros(len(indices), dtype = 'i8')
    #    for i,j in enumerate(indices):
    #        res[i] = arr[j]
    #    return res
    
    #cpdef double_select(self, arr, cond):
    #    cdef cnp.ndarray[long] indices
    #    cdef cnp.ndarray[double, cast = True] res
    #    cdef list lst
    #    cdef long i,j
    #    lst = []
    #    indices = np.where(self.to_bool_array(cond))[0]
    #    res = np.zeros(len(indices), dtype = 'f8')
    #    for i,j in enumerate(indices):
    #        res[i] = arr[j]
    #    return res
    
    #cpdef double_select_2D(self, arr, cond):
    #    cdef cnp.ndarray[long] indices
    #    cdef cnp.ndarray[double, ndim = 2, cast = True] res
    #    cdef list lst
    #    cdef long i,j
    #    lst = []
    #    indices = np.where(self.to_bool_array(cond))[0]
    #    res = np.zeros(np.shape(arr), dtype = 'f8')
    #    for i,j in enumerate(indices):
    #        res[i,:] = np.asarray(arr[j])
    #    return res
        
    cpdef void set_current_group(self, long i):
        self._current_group = i
        return
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_group_ledger(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.group)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_subgroup_ledger(self):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.subgroup)
        return res
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_visited(self):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.visited)
        return res
    #cpdef cnp.ndarray[long, ndim = 1, cast = True] get_surface_indices(self):
    #    cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.surface_indices)
    #    return res
    
    #cpdef cnp.ndarray[long, ndim = 1, cast = True] get_surface_ranks(self):
    #    cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.asarray(self.surface_rank)
    #    return res
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_group_mask(self):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.group_mask)
        return res
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_surface_mask(self):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.surface_mask)
        return res
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_subsurface_mask(self):
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.subsurface_mask)
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_group_particles(self, i):
        cdef cnp.ndarray[long, ndim = 1, cast = True] res = np.where(np.asarray(self.group) == i)[0]
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_current_group_particles(self):
        cdef list res = []
        cdef cpp_pq group_queue_copy
        cdef cpair[cnp.double_t, long] elem
        group_queue_copy = self.group_queue
        while not group_queue_copy.empty():
            elem = group_queue_copy.top()
            group_queue_copy.pop()
            res += [elem.second,]
        return np.array(res, dtype = long)
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_current_surface_particles(self):
        cdef list res = []
        cdef cpp_pq surface_queue_copy
        cdef cpair[cnp.double_t, long] elem
        surface_queue_copy = self.surface_queue
        while not surface_queue_copy.empty():
            elem = surface_queue_copy.top()
            surface_queue_copy.pop()
            res += [elem.second,]
        return np.array(res, dtype = long)
    
    cpdef list get_subgroups(self, i):
        cdef list res
        cdef cnp.ndarray[long, ndim = 1, cast = True] ids = self.get_group_particles(i)
        cdef cnp.ndarray[long, ndim = 1, cast = True] subgroups = self.get_subgroup_ledger()[ids]
        cdef long index, j
        res = [ids[subgroups == j] for j in range(0,int(np.max(subgroups))+1)]
        return res

    cpdef cnp.double_t get_long_range_fac(self):
        return self._long_range_fac
    
    #####=================== Boost!!! ===========================
    cpdef void set_x0(self, long i0, cnp.double_t[:] x0):
        '''
        Function which sets 'x0' the reference position for the calculation of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference position vector.

        '''
        self.i0 = i0
        self.x0 = x0
        cdef long i
        cdef cpair[cnp.double_t, long] elem
           
        while not self._computed_queue.empty():
            elem = self._computed_queue.top()
            self._computed_queue.pop()
            i = elem.second
            self._phi[i] = 0.
            self._computed[i] = False
            
        self._computed_queue = cpp_pq(compare_first)
        return
    
    cpdef void set_acc0(self, cnp.double_t[:] acc0):
        '''
        Function which sets 'acc0' the reference acceleration for the calculation of the boosted potential.
        
        Parameters:
        ----------
        acc0: (array of d-floats) d-dimensional refference acceleration vector.

        '''
        cdef int k
        self.acc0 = np.zeros(len(acc0))
        for k in range(len(acc0)):
            self.acc0[k] =  acc0[k]
        cdef long i
        cdef cpair[cnp.double_t, long] elem
           
        while not self._computed_queue.empty():
            elem = self._computed_queue.top()
            self._computed_queue.pop()
            i = elem.second
            self._phi[i] = 0.
            self._computed[i] = False
            
        self._computed_queue = cpp_pq(compare_first)
        return
    
    def set_fof_ids(self, ids_fof):
        '''
        Function which sets 'ids_fof' the ids of a precomputed group (not necesarrily FoF) used to find the first minimum of the boosted potential.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.

        '''
        self.ids_fof = ids_fof 
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
    
    cdef cnp.double_t[:] recentre_positions(self, cnp.double_t[:] pos, cnp.double_t[:] x0):
        '''
        Function which recentres and wraps the position vector 'pos' to a new centre at 'x0'.
        
        Parameters:
        ----------
        x0: (array of d-floats) d-dimensional refference acceleration vector.
        pos: (array of d-floats) positions of of particles in d-dimensional space.

        '''
        cdef cnp.double_t[:] res = np.zeros(np.shape(pos))
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
            
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx, temp_xa
        cdef int k
        cdef cnp.double_t res
        cdef cpair[cnp.double_t, long] elem
        
        if self._computed[i]:
            res = self._phi[i]
        else:
            x = self.recentre_positions(self.pos[i,:],self.x0)
            
            temp_xx = 0.0
            temp_xa = 0.0
            for k in range(len(x)):
                temp_xx += x[k]*x[k]
                temp_xa += x[k]*self.acc0[k]
            self._phi[i] = (self.pot[i] - self.pot[self.i0]) / self._scale_factor \
                            + temp_xa / self._scale_factor \
                            - self._long_range_fac * temp_xx  * self._scale_factor * self._scale_factor
            res = self._phi[i]
            
            self._computed[i] = True
            elem = (res, i)
            self._computed_queue.push(elem)
        return res

    cdef cnp.double_t phi_boost_physical(self, long i):
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
            
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx, temp_xa
        cdef int k
        cdef cnp.double_t res
        #cdef cpair[cnp.double_t, long] elem
        
        #if self._computed[i]:
        #    res = self._phi[i]
        #else:
        x = self.recentre_positions(self.pos[i,:],self.x0)

        temp_xx = 0.0
        temp_xa = 0.0
        for k in range(len(x)):
            temp_xx += x[k] * x[k]
            temp_xa += x[k] * self.acc0[k]
        
        res = self.pot[i] * self._scale_factor * self._scale_factor  + temp_xa - self._long_range_fac * temp_xx * self._scale_factor * self._scale_factor 
        
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
        
        indices = np.array(indices)
        if indices.size == 1:
            indices = np.array([indices.item(),])
        res = np.zeros(indices.size, dtype = 'f8')
        cond = np.zeros(indices.size, dtype = bool)#self.to_bool_array(self._computed)[indices]
        #if self._computed[i]:
        #    res = self._phi[i]
        res[cond] = np.asarray(self._phi)[indices[cond]]
        #x = np.zeros(len(self.pos[0]))
        for j,i in enumerate(indices[np.logical_not(cond)]):
            res[j] = self.phi_boost(i)
            #x = self.recentre_positions(self.pos[i,:],self.x0)
            #self._computed[i] = True
            #temp_xx = 0.0
            #temp_xa = 0.0
            #for k in range(len(x)):
            #    temp_xx += x[k]*x[k]
            #    temp_xa += x[k]*self.acc0[k]
            #res[j] = self.pot[i] * self._scale_factor * self._scale_factor  + temp_xa - self._long_range_fac * temp_xx
            #res = self._phi[i]
        if res.size == 1:
            return res.item()
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
    
    # ================== Particle Assignment ====================
    
    cpdef long first_minimum(self, long i0, double r = 1):
        cdef int counter = 1
        cdef long i = i0
        cdef long j, k
        cdef cnp.double_t dist = 0.0
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx = 0.0
        cdef cnp.double_t[:] phi_ngbs
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
                x = self.recentre_positions(self.pos[i], self.pos[i0])
                temp_xx = 0.0
                for k in range(len(x)):
                    temp_xx += x[k]*x[k]
                dist = np.sqrt(temp_xx)
                if dist > r:
                    if self.verbose: print(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.", flush = True)
                    self._too_far = True
                    return -1
                if counter > 1000:
                    if self.verbose: print("The minimum was not found after 1000 steps.\nThis may indicate that there is either no miminum or that you are starting index is too far away.", flush = True)
                    self._too_far = True
                    return -1
        x = self.recentre_positions(self.pos[i], self.pos[i0])
        temp_xx = 0.0
        for k in range(len(x)):
             temp_xx += x[k]*x[k]
        if dist > r:
            if self.verbose: print(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                  \nThis may indicate that there is either no miminum or that you are starting index is too far away.", flush = True)
            self._too_far = True
            return -1
        return i
    
    cpdef long fof_minimum(self, long[:] ids_fof):
        cdef long k, i_min
        cdef cnp.double_t phi_min
        cdef cpair[cnp.double_t, long] elem
        cdef cpp_pq loc_queue = cpp_pq(compare_first)
        for k in ids_fof:
            elem = (self.phi_boost(k), k)
            
            loc_queue.push(elem)
        elem = loc_queue.top()
        return elem.second
    
    cpdef long itt_minimum(self, long i0, cnp.double_t r = 1.):
        cdef long i0_itt = i0
        cdef long i0_fof, i0_temp
        cdef cbool min_found = False
        cdef set mem = set()

        if len(self.ids_fof) > 0:
            i0_fof = self.fof_minimum(self.ids_fof)
            i0_fof = self.first_minimum(i0_fof, r)
            if i0_fof not in set(self.ids_fof) or self._too_far:
                if self.verbose: print(f"The initial minimum is outside of the seed group falling back to standard approach.", flush = True)
                if self._too_far: self._too_far = False
                i0_temp = self.first_minimum(i0_itt, r)
                if i0_temp not in set(self.ids_fof) or self._too_far:
                    if self.verbose: print(f"The initial minimum is still outside of the seed group. Exiting", flush = True)
                    return -1
            else:
                i0_itt = i0_fof
        else:
            i0_itt = self.first_minimum(i0, r)

        self.set_x0(i0_itt, self.pos[i0_itt])
        if i0_itt == self.first_minimum(i0_itt, r):
            min_found = True
            i0 = i0_itt

        while not min_found: # Itterate over first minimum to check that it is indeed a minimum in it's own reference frame
            i0_temp = self.first_minimum(i0_itt, r)
            if i0_temp == -1 or self._too_far:
                if self.verbose: print(f"Something went wrong itterating over minima. Exiting", flush = True)
                return -1
            else: i0_itt = i0_temp
            if len(self.ids_fof) > 0: # If we have a fof seed make sure we don't leave the group through here.
                if i0_itt not in set(self.ids_fof) or self._too_far:
                    if self.verbose: print(f"The initial minimum is still outside of the seed group. Exiting", flush = True)
                    return -1

            self.set_x0(i0_itt, self.pos[i0_itt])
            if i0_itt == self.first_minimum(i0_itt, r): # This particle is the minimum in it's own refference frame.
                min_found = True
                i0 = i0_itt

            if i0_itt in mem: # If we end up here we're in a multiple particle loop
                if self.verbose: print(f"Stuck in a loop finding first minimum.\n{mem}\n Exiting", flush = True)
                return -1
            mem.add(i0_itt) # Keep track of which particles have been visited to avoid getting stuck in a loop
        return i0
    
    cpdef void fill_below(self, cnp.int64_t i):
        '''
        Itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle i. 
        
        Note that the particles are saved within a set 'self.i_in' that is saved internally to the class and all the direct neighbours to the particles in this set are saved to a second set 'self.i_surf'. 
        
        These sets should not be modified directly as this can cause issues with the segmentation algorithm.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        '''
        
        cdef double phi_curr = self.phi_boost(i)
        cdef long[:] ngbs_temp
        cdef long j, k, l, index
        cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.ndarray[long, ndim = 1, cast = True] indices
        cdef cpair[cnp.double_t, long] elem, top_elem
        
        
        
        # Initialise queues
        self.group_queue = cpp_pq(compare_first)
        self.surface_queue = cpp_pq(compare_first)
        
        # Put starting particle in group
        self.group[i] = self._current_group
        self.group_mask[i] = True
        self.visited[i] = True
        elem = (self.phi_boost(i), i)
        self.group_queue.push(elem)
        self._current_group_size += 1
        
        # Calculate it's potential and define its neighbours as surface particles
        
        self._current_surface_size = 0
        ngbs_temp = self.ngbs[i]
        for k in ngbs_temp:
            if not self.visited[k]:
                self._current_surface_size += 1
                self.surface_mask[k] = True
                elem = (self.phi_boost(k), k)
                self.surface_queue.push(elem)
        
        # Take surface particle with lowest potential
        
        elem = self.surface_queue.top()
        self.surface_queue.pop()
        self._current_surface_size -= 1
        j = elem.second
        self.surface_mask[j] = False
        
        while self.phi_boost(j) - phi_curr < 0.0:
            self.group[j] = self._current_group
            self.group_mask[j] = True
            self._current_group_size +=1
            self.visited[j] = True
            
            elem = (self.phi_boost(j), j)
            self.group_queue.push(elem)
            self._current_group_size += 1
            
            ngbs_temp = self.ngbs[j]
            
            for k in ngbs_temp:
                if self.surface_mask[k] or self.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    elem = (self.phi_boost(k), k)
                    self.surface_queue.push(elem)
                    self._current_surface_size += 1
                    self.surface_mask[k] = True

            top_elem = self.surface_queue.top()
            self.surface_queue.pop()
            self._current_surface_size -= 1
            j = top_elem.second
            self.surface_mask[j] = False
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
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx
        cdef double dist
        cdef double phi_curr = self.phi_boost(i)
        cdef long j, k, index, j_old
        cdef long[:] ngbs_temp
        cdef int counter = 1
        cdef double phi_j
        
        # Empty queue
        cdef cpair[cnp.double_t, long] elem
        
        if not self.subsurface_queue.empty():
            #print(f"reset queue", flush = True)
            self.subsurface_queue = cpp_pq(compare_first)
            self._current_subsurface_size = 0
            
        if not self.subgroup_queue.empty():
            #print(f"reset queue", flush = True)
            self.subgroup_queue = cpp_pq(compare_first)
            self._current_subgroup_size = 0
        
        # Start by adding the refference particle to the group
        self.subgroup_mask[i] = True
        self.visited[i] = True
        elem = (self.phi_boost(i),i)
        self.subgroup_queue.push(elem)
        self._current_subgroup_size += 1
        # Calculate it's potential and define its neighbours as surface particles
        
        ngbs_temp = self.ngbs[i]

        
        for k in ngbs_temp:
            if not self.visited[k]:
                self.subsurface_mask[k] = True
                elem = (self.phi_boost(k), k)
                self.subsurface_queue.push(elem)
                self._current_subsurface_size += 1
        
        if self.subsurface_queue.empty():
            #self._too_far = True
            if self.verbose: 
                print(f"subsurface queue started empty {i}", flush = True)
                print(f"ngbs: {[ngb for ngb in self.ngbs[i]]}", flush = True)
                print(f"pots: {[self.phi_boost(ngb) for ngb in self.ngbs[i]]}", flush = True)
                print(f"visited: {[self.visited[ngb] for ngb in self.ngbs[i]]}", flush = True)
                print(f"groups: {[self.group[ngb] for ngb in self.ngbs[i]]}", flush = True)
            return
        
        elem = self.subsurface_queue.top()
        j = elem.second
        j_old = -1
        
        while self.phi_boost(j) - phi_curr < 0:
            #========= Test ================ 
            if self.visited[j]: # If we've already put j in the subgroup or is part of the main group
                if self.verbose: print(f"Particle {j} already visited but in subsurface queue...", flush = True)
                self.subsurface_queue.pop()
                self.subsurface_mask[j] = False
                self._current_subsurface_size -= 1
                if self.subsurface_queue.empty():
                    #self._too_far = True
                    if self.verbose: print(f"subsurface queue empty {j}", flush = True)
                    return
                elem = self.subsurface_queue.top()
                j = elem.second
                continue
            
            
            if j == j_old:
                self._too_far = True
                if self.verbose: print(f"ya in a loop it seems {j}", flush = True)
                return
            j_old = j
            #=============================== 
            #print(j, end = ' ', flush = True)
            counter += 1        
            # By taking this one we ensure we are always going down the lowest brach first.
            phi_j = self.phi_boost(j)
            
            if counter % 100 == 0: # Check if we haven't wandered too far away
                x = self.recentre_positions(self.pos[j], self.x0)
                temp_xx = 0.0
                for k in range(len(x)):
                    temp_xx += x[k]*x[k]
                dist = np.sqrt(temp_xx)
                if dist > 3*self.max_dist:
                    if self.verbose:
                        print(f"The new minimum is {float(dist):.3} (> {float(3*self.max_dist):.3}) Mpc away from the starting position.\nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                    self._too_far = True
                    return
    
                if self._current_subgroup_size > self._current_group_size:
                    #raise RecursionError(f"Trying to add in a structure which is much larger than the current structure. Exiting.")
                    if self.verbose:
                        print(f"Trying to add in a structure which is much larger than the current structure {self._current_subgroup_size} > {self._current_group_size}. Exiting.")
                    self._too_far = True
                    return
            
            # If we are in this we already know that j has a lower potential
            # So we can add it directly and remove it from the surface
            self.subsurface_queue.pop()
            self.subsurface_mask[j] = False
            self._current_subsurface_size -= 1
            #if self.visited[j]:
            #    continue
            self.subgroup_mask[j] = True
            self.visited[j] = True
            self._current_subgroup_size += 1
            self.subgroup_queue.push(elem)
            
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
                    elem = (self.phi_boost(k), k)
                    self.subsurface_queue.push(elem)
                    self._current_subsurface_size += 1
                    self.subsurface_mask[k] = True
            #========= Test ================        
            if self.subsurface_queue.empty():
                #self._too_far = True
                if self.verbose: print(f"subsurface queue empty {j}", flush = True)
                return
            #=============================== 
            # Take subsurface particle with lowest potential
            elem = self.subsurface_queue.top()
            j = elem.second
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
        cdef long j, k, index, elem, k_loc
        
        cdef long i_cons
        cdef cnp.double_t[:] phi_boost_ngbs
        cdef cnp.ndarray[long, ndim = 1, cast = True] group_particles
        cdef long[:] ngbs_temp
        cdef cbool low_pot_cond
        cdef cbool if_cond, size_cond, first
        cdef double start_loop, end_loop, start, end
        
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx = 0.0
        cdef cnp.double_t dist, phi_cons, phi_k
        cdef cpair[cnp.double_t, long] qelem
        cdef long i_min, i_max, i
        
        group_particles = self.get_current_group_particles()
        n_part = group_particles.size
        phi_min = self.phi_boost(group_particles[0])
        i_min = group_particles[0]
        phi_max = self.phi_boost(group_particles[0])
        i_max = group_particles[0]
        
        for k in group_particles:
            x = self.recentre_positions(self.pos[k], self.x0)
            temp_xx = 0.0
            for k_loc in range(len(x)):
                temp_xx += x[k_loc]*x[k_loc] 
            dist = np.sqrt(temp_xx)
            if dist > self.max_dist:
                self.max_dist = dist
            phi_k = self.phi_boost(k)
            if phi_k < phi_min:
                phi_min = phi_k
                i_min = k
            if phi_k > phi_max:
                phi_max = phi_k
                i_max = k
                
        i_prev = -1

        for _ in range(len(self.pot)): # <== I just don't want to write 'while True:'
            
            if self.surface_queue.empty():
                if self.verbose: print('Surface empty', flush = True)
                break
            
            qelem = self.surface_queue.top()
            phi_cons = qelem.first
            i_cons = qelem.second
            
            if i_cons == i_prev:
                raise RecursionError(f"The algorithm seems to have gotten stuck at i = {i_cons} with {self._current_group_size} particles assigned and {self._current_surface_size} pending...")
            i_prev = i_cons
            
            if self.visited[i_cons]: # If we've already put i_cons in the group but it was still in the surface.
                if self.verbose: print(f"{i_cons} had already been visited but was still in surface queue")
                self.surface_queue.pop()
                self.surface_mask[i_cons] = False
                self._current_surface_size -= 1
                continue
                
            self.visited[i_cons] = True
            ngbs_temp = self.ngbs[i_cons]
            
            low_pot_cond = False
            for j in range(len(ngbs_temp)):
                if self.phi_boost(ngbs_temp[j]) < self.phi_boost(i_cons):
                    low_pot_cond = True
                    break

            x = self.recentre_positions(self.pos[i_cons], self.x0)
            temp_xx = 0.0
            for k_loc in range(len(x)):
                temp_xx += x[k_loc]*x[k_loc] 
            dist = np.sqrt(temp_xx)
            if dist > self.max_dist:
                self.max_dist = dist
            
            if not low_pot_cond:
                # This can happen when hitting boundaries, or for very small local minima
                # These particles should be recaptured by the code later if we just skip them.
                if self.verbose: 
                    print(f"{i_cons} low_pot_cond failed: Has no neighbours with lower potential", flush = True)
                    print(f"Relative potentials: {[self.phi_boost(ngb) - self.phi_boost(i_cons) for ngb in self.ngbs[i_cons]]}", flush = True)
                    print(f"Neighbouring groups: {[self.group[ngb] for ngb in self.ngbs[i_cons]]}", flush = True)
                    print(f"Neighbouring visits: {[self.visited[ngb] for ngb in self.ngbs[i_cons]]}", flush = True)
                    print(f"Neighbouring surface: {[self.surface_mask[ngb] for ngb in self.ngbs[i_cons]]}", flush = True)
                self.surface_queue.pop()
                self._current_surface_size -= 1
                self.surface_mask[i_cons] = False
                self.visited[i_cons] = False
                #self._computed[i_cons] = False
                continue
            
            if_cond = True
            
            for k, elem in enumerate(ngbs_temp):
                if self.phi_boost(elem) < self.phi_boost(i_cons):
                    if_cond *= self.group_mask[elem]
            
            if if_cond:
                # Tag i_cons as part of the main group
                #print(f"mg {i_cons} |", end = " ")
                self.group[i_cons] = self._current_group
                self.group_mask[i_cons] = True
                self._current_group_size += 1
                qelem = (phi_cons, i_cons)
                self.group_queue.push(qelem)
                
                # Remove it from the surface
                self.surface_queue.pop()
                self._current_surface_size -= 1
                self.surface_mask[i_cons] = False
                
                if phi_cons < phi_min:
                    phi_min = phi_cons
                    i_min = i_cons
                if phi_cons > phi_max:
                    phi_max = phi_cons
                    i_max = i_cons

                for k in ngbs_temp:
                    if self.surface_mask[k] or self.visited[k]:
                        # Avoid duplicates or going back to a particle that has already been visited
                        continue
                    else:
                    
                        # Insert particle in sorted order
                        self.surface_mask[k] = True
                        qelem = (self.phi_boost(k), k)
                        self.surface_queue.push(qelem)
                        self._current_surface_size += 1
                

            else:
                # Or check if there is a substructure
                self.visited[i_cons] = False
                self.new_min = phi_cons
                
                #self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)     
                #self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
                self.subsurface_queue = cpp_pq(compare_first)
                self.subgroup_queue = cpp_pq(compare_first)
                self._current_subsurface_size = 0
                self._current_subgroup_size = 0
                #try:
                self.fill_below_substructure(i_cons, phi_min)
                #except RecursionError:
                if self._too_far:
                    # Traveled too far or adding in a large group.
                    if self.verbose: print('Traveled too far exitting... ', self._current_subgroup_size, end = ' ', flush = True)
                    while not self.subgroup_queue.empty():
                        qelem = self.subgroup_queue.top()
                        self.subgroup_queue.pop()
                        k = qelem.second
                        self.visited[k] = False
                        self.subgroup_mask[k] = False
                        
                    
                    while not self.subsurface_queue.empty():
                        qelem = self.subsurface_queue.top()
                        self.subsurface_queue.pop()
                        k = qelem.second
                        self.visited[k] = False
                        self.subsurface_mask[k] = False
                    self._too_far = False
                    break
                
                if self.new_min <= phi_min:
                    # Moved into a lower potential well => exit
                    if self.verbose: print('Found lower minimum:', self._current_subgroup_size, i_cons, end = ' ', flush = True)
                    while not self.subgroup_queue.empty():
                        qelem = self.subgroup_queue.top()
                        self.subgroup_queue.pop()
                        k = qelem.second
                        self.visited[k] = False
                    
                    while not self.subsurface_queue.empty():
                        qelem = self.subsurface_queue.top()
                        self.subsurface_queue.pop()
                        k = qelem.second
                        self.visited[k] = False
                    break
                    
                
                else:
                    # Found a structure => add it in 
                    # Merge surfaces
                    self.visited[i_cons] = True 
                    # Remove i_cons from sufrace
                    self.surface_queue.pop()
                    self._current_surface_size -= 1
                    self.surface_mask[i_cons] = False
                    #first = True
                    while not self.subsurface_queue.empty():
                        qelem = self.subsurface_queue.top()
                        self.subsurface_queue.pop()
                        k = qelem.second
                        self.subsurface_mask[k] = False
                        #print(f"ss {k} |", end = " ")
                        if self.surface_mask[k] or self.visited[k]:
                            # Avoid duplicates or going back to a particle that has already been visited
                            continue
                        else:
                            self.surface_mask[k] = True
                            self.surface_queue.push(qelem)
                            self._current_surface_size += 1
                    self._current_subsurface_size = 0    
                    
                    # Merge Groups
                    size_cond = self._current_subgroup_size > 20
                    if size_cond:
                        self._current_subgroup += 1
                    while not self.subgroup_queue.empty():

                        qelem = self.subgroup_queue.top()
                        self.subgroup_queue.pop()
                        k = qelem.second
                        phi_k = qelem.first
                        #print(f"sg {k} |", end= " ")
                        self.visited[k] = True
                        self.group[k] = self._current_group
                        self.group_mask[k] = True
                        self.subgroup_mask[k] = False
                        self._current_group_size += 1
                        self.group_queue.push(qelem)
                        
                        if size_cond:
                            self.subgroup[k] = self._current_subgroup
                        x = self.recentre_positions(self.pos[k], self.x0)
                        temp_xx = 0.0
                        for k_loc in range(len(x)):
                            temp_xx += x[k_loc]*x[k_loc] 
                        dist = np.sqrt(temp_xx)
                        if dist > self.max_dist:
                            self.max_dist = dist
                        
                        # Keep min and max up to date
                        if phi_k < phi_min:
                            phi_min = phi_k
                            i_min = k
                        if phi_k > phi_max:
                            phi_max = phi_k
                            i_max = k      
                    self._current_subgroup_size = 0 
                if self._current_group_size > 0.25*len(self.pot):
                    # Temporary measure to get a catalogue in EdS-cond
                    if self.verbose: print('Reached size limit (npart/4):', self._current_group_size, end = ' ', flush = True)
                    break

        
        return i_min, i_max
    
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
        
        cdef cnp.double_t[:] v_mean 
        cdef cnp.double_t[:,:] v_in
        cdef cnp.double_t[:] v
        cdef cnp.double_t[:] x
        cdef cnp.double_t[:] phi_p
        cdef cnp.double_t[:] K
        cdef cnp.double_t[:] E
        cdef cnp.uint8_t[:] bound_mask
        cdef double phi_p_sad, temp_xx, temp_xv, temp_vv, factor_xx_K, factor_xx_phi, factor_xv, sqrt_a
        cdef long i = 0
        cdef long index
        cdef int j = 0
        
        i_in_arr = self.get_current_group_particles()#np.where(self.group_mask)[0]
        
        v_in = np.zeros((len(i_in_arr), self.vel.shape[1]))
        K = np.zeros(len(i_in_arr))
        bound_mask = np.zeros(len(i_in_arr), dtype = bool)
        phi_p = np.zeros(len(i_in_arr))
        E = np.zeros(len(i_in_arr)) 
        v_mean = np.zeros(self.vel.shape[1])
        
        v = np.zeros(self.vel.shape[1])
        for j,i in enumerate(i_in_arr):
            v_in[j,:] = self.vel[i]
        
        for j in range(self.vel.shape[1]):
            v_mean[j] = np.mean(v_in[:,j]) # remove the mean velocity of the group
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
            
        factor_xx_K = 0.5*self._Ha**2 * self._scale_factor*self._scale_factor
        factor_xx_phi = 0.25*self.Omega_m(1.)*self._H0**2 / self._scale_factor
        factor_xv = self._scale_factor * self._Ha
        
        # phi_p_sad = self.phi_boost_physical(i_sad, v_mean, use_vel = False) + factor_xx_phi * temp_xx
        # phi_p_min = self.phi_boost_physical(i_min, v_mean, use_vel = False) #+ factor_xx_phi * temp_xx
        
        phi_p_sad = self.phi_boost(i_sad) + factor_xx_phi * temp_xx
        phi_p_min = self.phi_boost(i_min) #+ factor_xx_phi * temp_xx
        
        sqrt_a = np.sqrt(self._scale_factor)
        #phi_p_sad = self.phi_boost(i_sad)
        for i, index in enumerate(i_in_arr):
            x = self.recentre_positions(self.pos[index],self.pos[i_min])
            for j in range(len(x)):
                v[j] = (v_in[i,j] - v_mean[j]) # These should be comoving velocities
            temp_xx = 0.0
            temp_xv = 0.0
            temp_vv = 0.0
            for j in range(len(x)):
                temp_xx += x[j]*x[j]
                temp_xv += v[j]*x[j]
                temp_vv += v[j]*v[j]
            
            
            K[i] = 0.5 * temp_vv + factor_xv * temp_xv + factor_xx_K * temp_xx 
            #phi_p[i] = self.phi_boost_physical(index, v_mean) + factor_xx_phi * temp_xx 
            phi_p[i] = self.phi_boost(index) + factor_xx_phi * temp_xx + self._long_range_fac * temp_xx * self._scale_factor * self._scale_factor
            #K[i] = 0.5 * temp_vv 
            #phi_p[i] = self.phi_boost(index)
            E[i] = K[i] + phi_p[i] - phi_p_min # <= Converted to physical potential
            if E[i] < phi_p_sad - phi_p_min:
                bound_mask[i] = True
        
        return bound_mask

    
    cpdef segment(self, long i0, cnp.double_t[:] acc0, cnp.double_t r = 1.):
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
        cdef long i_min, i_sad, i0_fof, i0_temp, i0_itt
        #cdef cpp_pq dist_queue
        #cdef cpair[cnp.double_t, long] elem
        cdef set mem = set()
        cdef cbool min_found
        #self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        self.set_acc0(acc0)
        self.set_x0(i0, self.pos[i0])
        
        self._current_group += 1
        self._current_subgroup = 0
        # Verify that the first minimum is not too far away.
        
        i0 = self.itt_minimum(i0,r)
        if i0 == -1:
            return np.array([], dtype = long), i0, i0
        #if len(self.ids_fof) > 0:
        #    i0 = self.fof_minimum(self.ids_fof)
        #else:
        #    i0 = self.first_minimum(i0, r)
        
        # Find all particles with potential lower than minimum (This should only give back 1 particle)
        self.set_x0(i0, self.pos[i0])
 
        self.fill_below(i0)
        # Grow potential surface
        i_min, i_sad = self.grow()
        #self.visited = np.zeros(self.pot.size, dtype = bool) # For now we just reset the list
        #Binding check
        if self.no_binding: 
            return self.get_current_group_particles(), i_min, i_sad
        bound_mask = self.is_bound(i_min, i_sad)
        return self.get_current_group_particles()[bound_mask], i_min, i_sad # output: i_min, i_sad, n_part(, subs, i_in)
    
    cpdef void reset(self):
        self.i0 = -1
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(self.pot.size, dtype = bool) # We may want to review these assignements
        
        self._computed = np.zeros(self.pot.size, dtype = bool)
        self._phi = np.zeros(self.pot.size, dtype = 'f8') 
        self.group = np.zeros(self.pot.size, dtype = 'i8')
        self.subgroup = np.zeros(self.pot.size, dtype = 'i8')

        
        self.group_mask = np.zeros(self.pot.size, dtype = bool)
        self.surface_mask = np.zeros(self.pot.size, dtype = bool)
        self.surface_queue = cpp_pq(compare_first)
        self.group_queue = cpp_pq(compare_first)

        
        self.subgroup_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_mask = np.zeros(self.pot.size, dtype = bool)
        self.subsurface_queue = cpp_pq(compare_first)
        self.subgroup_queue = cpp_pq(compare_first)
        
        self.bound_mask = np.zeros(self.pot.size, dtype = bool)
        
        self._current_group = 0
        self._current_subgroup = 0
        self._current_group_size = 0
        self._current_subgroup_size = 0
        self._current_surface_size = 0
        self._current_subsurface_size = 0
        
        self.new_min = 3.4e38
        return
    
    cdef void _reset_arrays(self, long i):
            self.visited[i] = False # We may want to review these assignements

            self._computed[i] = False
            self._phi[i] = 0.
            self.group[i] = 0
            self.subgroup[i] = 0

            self.group_mask[i] = False
            self.surface_mask[i] = False

            self.subgroup_mask[i] = False
            self.subsurface_mask[i] = False
            
            self.bound_mask[i] = False
            return
    
    cpdef void reset_computed_particles(self):
        self.i0 = -1
        self.x0 = None
        self.acc0 = None
        #cdef long[:] group_ids = self.get_current_group_particles()
        #cdef long[:] surface_ids = self.get_current_surface_particles()
        #cdef long[:] ids = np.hstack([group_ids,surface_ids])
        cdef long i
        cdef cpair[cnp.double_t, long] elem
           
        while not self._computed_queue.empty():
            elem = self._computed_queue.top()
            self._computed_queue.pop()
            i = elem.second
            self._reset_arrays(i)
            
        self._computed_queue = cpp_pq(compare_first)
        self.surface_queue = cpp_pq(compare_first)
        self.group_queue = cpp_pq(compare_first)
        self.subsurface_queue = cpp_pq(compare_first)
        self.subgroup_queue = cpp_pq(compare_first)
        
        self._current_group = 0
        self._current_subgroup = 0
        self._current_group_size = 0
        self._current_subgroup_size = 0
        self._current_surface_size = 0
        self._current_subsurface_size = 0
        
        self.new_min = 3.4e38
        return