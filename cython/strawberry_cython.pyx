# distutils: language = c++
import numpy as np
cimport numpy as cnp
from scipy.integrate import quad
from libcpp cimport bool as cbool
from libcpp.pair cimport pair as cpair

cdef extern from "cpp_priority_queue.hpp":
    cdef cppclass cpp_pq:
        cpp_pq(...) except +
        void push(cpair[cnp.double_t,long])
        cpair[cnp.double_t,long] top()
        void pop()
        cbool empty()
        
cdef cbool compare_first(cpair[cnp.double_t, long] a, cpair[cnp.double_t, long] b):
    return a.first > b.first #flipping this sign says in which order we want the queue "< largest first" , "> smallest first"

cdef class Tracker:
    '''
    Tracker class, used internally to keep track of groups of particles. Composed of a priority queue of c++ pairs (potential, id) sorted in increasing potential. Only highest potential particle can be accessed from the queue directly. To access other particles one must return the full queue using get_particles.

    Parameters
    ---------
    nparts: (int) Total number of particles in the simulation

    Methods:
    ---------
    - add_elem: Add element to tracker
    
    - get_top_elem: Get top element of tracker
    
    - remove_top_elem: Delete top element of the tracker
    
    - get_queue: Get copy of internal priority queue
    
    - get_size: Get current size of queue
    
    - is_empty: Check if queue is empty
    
    - is_mask: Check if a given id is within the queue
    
    - get_particles: Returns list of ids of all particles in the queue
    '''
    cdef cpp_pq queue
    cdef cnp.uint8_t[:] mask
    cdef long size
    cdef long _npart_sim

    def __init__(self, long npart):
        self._npart_sim = npart
        self.mask = np.zeros(self._npart_sim, dtype = bool)
        self.size = 0
        self.queue = cpp_pq(compare_first)
        return

    cdef void add_elem(self, cpair[cnp.double_t,long] elem):
        '''
        Add element (float64, int64) to the tracker.
        
        Parameters:
        ---------
        elem: (cpair float64, int64) Tracker element to be added
        '''
        self.size += 1
        self.mask[elem.second] = True
        self.queue.push(elem)
        return

    cdef cpair[cnp.double_t,long] get_top_elem(self):
        '''
        Return top element (float64, int64) of the tracker. Data is accessed using elem.first and elem.second properties
        
        Outputs:
        ---------
        elem: (cpair float64, int64) Tracker element to be added
        '''
        return self.queue.top()
    
    cdef void remove_top_elem(self):
        '''
        Removes top element of the tracker.
        '''
        cdef cpair[cnp.double_t,long] elem
        cdef long i
        elem = self.queue.top()
        self.queue.pop()
        self.size -= 1
        i = elem.second
        self.mask[i] = False
        return
    
    cdef cpp_pq get_queue(self):
        '''
        Returns a copy of the trackers internal priority queue.
        
        Outputs:
        ---------
        queue: (C++ priority queue) copy of internal priority queue
        '''
        return self.queue

    cdef cnp.int64_t get_size(self):
        '''
        Returns the current number of elements being tracked

        Outputs:
        ---------
        size: (int64) number of elements tracked by tracker
        '''
        return self.size

    cdef void reset(self):
        '''
        Resets tracker by iteratively removing all elements.
        '''
        while not self.queue.empty():
            self.remove_top_elem()
        return

    cdef cbool is_empty(self):
        '''
        Checks if tracker is empty

        Outputs:
        ---------
        is_empty: (bool) True if the are no elements being tracked
        '''
        return self.queue.empty()

    cdef cnp.uint8_t is_mask(self, cnp.int64_t i):
        '''
        Return state of the internal boolean mask of the tracker at index i. Allows to check if the particle with index i is being tracked.

        Parameters:
        ---------
        i: (int64) Particle index
        
        Outputs:
        ---------
        mask: (bool) Value of mask for particle i
        '''
        return self.mask[i]

    cdef cnp.ndarray[long, ndim = 1, cast = True] get_particles(self):
        '''
        Returns an array of the indices of all of the particles currently being tracked.

        Outputs:
        ---------
        res: (Array of int64) Array of the indices of all particles currently tracked by the tracker.
        '''
        cdef list res = []
        cdef cpp_pq queue_copy
        cdef cpair[cnp.double_t, long] elem
        queue_copy = self.queue
        while not queue_copy.empty():
            elem = queue_copy.top()
            queue_copy.pop()
            res += [elem.second,]
        return np.array(res, dtype = 'i8')

cdef class Halo:
    '''
    Halo class. Used to track halo properties as defined by the potential well.

    Parameters
    ---------
    nparts: (int) Total number of particles in the simulation
    ids_fof: (Array of ints) optional, array of particle indices used as an initial seed for the particle assignment procedure.

    Methods:
    ---------
    - reset: Applies a full reset to the Halo, deleting all saved information.
    
    -reset_computed_particles: Reset internal values for only computed particles, faster than reset() and avoid realocations
    
    -set_x0: Initialises i0 and x0 the index and position of the reference particle used for the calculation of the boosted potential
    
    -set_acc0: Initialises acc0 the reference acceleration used for the calculation of the boosted potential
    
    -get_i0: Returns i0 the index of the reference particle used for the calculation of the boosted potential
    
    -get_acc0: Returns acc0 the reference acceleration used for the calculation of the boosted potential
    
    -set_fof_ids: Sets the indices of the seed group used to find the reference position and acceleration
    
    -to_bool_array: Utility method used to transform uint8 MemoryViews into boolean NumPy Arrays
    
    -set_current_group: (To be deprecated) sets the id of the group
    
    -get_visited: Returns boolean array selecting all particles that have been visited (Memory warning for large simulations)
    
    -get_group_mask: Returns boolean array selecting all particles that have been assigned as being within the potential well of the halo (Memory warning for large simulations)
    
    -get_surface_mask: Returns boolean array selecting all particles that have been assigned as being connectied/adjacent to the potential well of the halo (Memory warning for large simulations)
    
    -get_current_group_particles: Returns array of indices of all particles assigned as being within the potential well of the halo
    
    -get_current_surface_particles: Returns array of indices of all particles assigned as being connectied/adjacent to the potential well of the halo
    
    -get_subgroups: Returns list of arrays containing the indices of particles assigned to a given subgroup
    
    -get_bound_mask: Returns boolean mask of the same size a sget_current_group_particles selecting particles which are dynamically bound tot the potential well
    '''
    
    cdef long nparts
    
    cdef long i0
    cdef long i_min
    cdef long i_max
    
    cdef cnp.double_t[:] x0
    cdef cnp.double_t[:] acc0
    cdef cnp.uint8_t[:] visited

    cdef cnp.double_t max_dist
    cdef cnp.double_t new_min
    
    cdef long[:] ids_fof
    
    cdef Tracker computed_tracker
    cdef Tracker group_tracker
    cdef Tracker subgroup_tracker
    cdef Tracker surface_tracker
    cdef Tracker subsurface_tracker
    cdef cnp.uint8_t[:] bound_mask
    cdef cnp.int64_t _current_group
    cdef cnp.int64_t _current_subgroup
    cdef cnp.double_t[:] _phi
    
    cdef list subgroups
    cdef cbool _too_far

    def __init__(self, long nparts, cnp.ndarray[long, ndim = 1] ids_fof = np.empty([0], dtype = 'i8')):
        
        self.nparts = nparts
        self.computed_tracker = Tracker(self.nparts)
        self.group_tracker = Tracker(self.nparts)
        self.subgroup_tracker = Tracker(self.nparts)
        self.surface_tracker = Tracker(self.nparts)
        self.subsurface_tracker = Tracker(self.nparts)
        
        self.ids_fof = ids_fof
        
        # self.bound_mask is left unallocated until the we know the size of the halo.
        
        self._current_group = 0
        self._current_subgroup = 0

        self.i0 = -1
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(self.nparts, dtype = bool) # We may want to review these assignements
        self._phi = np.zeros(self.nparts, dtype = 'f8')
        self.subgroups = []
        self.max_dist = 1. 
        # We should only realy be worried once structures start getting biger than 1 Mpc.
        # Note that this is a first guess and is changed adaptively to accomodate larger objects.
        self.new_min = 3.4e38
        self._too_far= False

        self.i_min = -1
        self.i_max = -1
        return
    
    cpdef void reset(self):
        '''
        Method to operate a full reset of internal variables. This can be quite slow for large simulations. In this case prefer, reset_computed_particles.
        '''
        # Operates a full reset of all the internal arrays and structures. Can be a bit slow for large simulations
        self.i0 = -1
        self.x0 = None
        self.acc0 = None
        self.visited = np.zeros(self.nparts, dtype = bool) 
        self._phi = np.zeros(self.nparts, dtype = 'f8') 
        self.computed_tracker = Tracker(self.nparts)
        self.group_tracker = Tracker(self.nparts)
        self.subgroup_tracker = Tracker(self.nparts)
        self.surface_tracker = Tracker(self.nparts)
        self.subsurface_tracker = Tracker(self.nparts)
        
        self._current_group = 0
        self._current_subgroup = 0
        
        self.new_min = 3.4e38
        self.subgroups = []
        self._too_far= False
        self.i_min = -1
        self.i_max = -1
        return
    
    cdef void _reset_arrays(self, long i):
        '''
        Method to reset the internal values for a single particle.
        This method is dangerous as it does not respect the order of the _computed_queue priority queue.
        Do not use unless stricly respecting this ordering.
        '''
        self.visited[i] = False 
        self._phi[i] = 0.
        
        return
    
    cpdef void reset_computed_particles(self):
        '''
        Method that reset all the internal values for all particles that have a computed boosted potentials and resets all internal trackers. Faster than reset but can be exposed to issues if the tracking is tampered.
        '''
        # Only resets particles that for which phi_boost has been computed at least once.

        cdef long i
        cdef cpair[cnp.double_t, long] elem

        while not self.computed_tracker.is_empty():
            elem = self.computed_tracker.get_top_elem()
            self.computed_tracker.remove_top_elem()
            i = elem.second
            self._reset_arrays(i)

        self.group_tracker.reset()
        self.subgroup_tracker.reset()
        self.surface_tracker.reset()
        self.subsurface_tracker.reset()
        
        self._current_group = 0
        self._current_subgroup = 0

        self.new_min = 3.4e38
        self.subgroups = []
        self._too_far= False
        self.i_min = -1
        self.i_max = -1
        return
    
    
    cpdef void set_x0(self, long i0, cnp.double_t[:] x0):
        '''
        Function which sets 'i0' and 'x0' the index and position of the reference particle for the calculation of the boosted potential. 
        While it is technically possible to use the index of one particle and the position of another this is not advised and will lead to undified behaviour
        
        Parameters:
        ----------
        i0: (int) index of reference particle
        x0: (array of d-floats) d-dimensional refference position vector.

        '''
        self.i0 = i0
        self.x0 = x0
           
        self.reset_computed_particles()
        return

    cpdef long get_i0(self):
        '''
        Returns index of reference particle used to grow the Halo
        
        Outputs:
        ----------
        i0: (int) index of reference particle
        '''
        return self.i0
    
    cpdef cnp.ndarray[cnp.float64_t, ndim = 1, cast = True] get_x0(self):
        '''
        Returns position of reference particle used to grow the Halo
        
        Outputs:
        ----------
        x0: (array of float) 1D array of reference position used to compute the boosted potential
        '''
        return np.asarray(self.x0)
    
    cpdef cnp.ndarray[cnp.float64_t, ndim = 1, cast = True] get_acc0(self):
        '''
        Returns reference acceleration used to compute de boosted potential

        Outputs:
        ----------
        acc0: (array of float) 1D array of reference acceleration used to compute the boosted potential
        '''
        return np.asarray(self.acc0)
    
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
        #cdef long i
        #cdef cpair[cnp.double_t, long] elem
           
        self.reset_computed_particles()
        return
    
    cpdef set_fof_ids(self, ids_fof):
        '''
        Function which sets 'ids_fof' the ids of a precomputed group (not necesarrily FoF) used to find the first minimum of the boosted potential.
        
        Parameters:
        ----------
        ids_fof: (array of int) Array holding the indices of all the particles in a prior group.

        '''
        self.ids_fof = ids_fof 
        return

    
    def to_bool_array(self, lst):
        '''
        Transforms cython uint8 MemoryView into numpy boolean array that is usable as a mask

        Parameters:
        ----------
        lst: 1-D MemoryView of dtype uint8

        Outputs:
        ----------
        res: 1-D NumPy array of dtype bool
        '''
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] res
        res = np.array(lst, dtype=bool)
        return res
        
    cpdef void set_current_group(self, long i):
        '''
        Set current group index (To be deprecated: Currently Unused)
        '''
        self._current_group = i
        return
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_visited(self):
        '''
        Returns 1D mask of size n-parts flagging particles that have been visited by the ParticleAssigner
        
        Outputs:
        ----------
        mask: (Array of bool) Boolean mask of all particles visited during the particle assignment procedure
        '''
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.visited)
        return res

    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_group_mask(self):
        '''
        Returns 1D mask of size n-parts flagging particles that have been assigned to the halo group by the ParticleAssigner
        
        Outputs:
        ----------
        mask: (Array of bool) Boolean mask of all particles assigned to the halo group during the particle assignment procedure
        '''
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.group_tracker.get_mask())
        return res
    
    cpdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] get_surface_mask(self):
        '''
        Returns 1D mask of size n-parts flagging particles that have been assigned to the halo surface by the ParticleAssigner
        
        Outputs:
        ----------
        mask: (Array of bool) Boolean mask of all particles assigned to the halo surface during the particle assignment procedure
        '''
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast = True] res = self.to_bool_array(self.surface_tracker.get_mask())
        return res
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_current_group_particles(self):
        '''
        Returns 1D array of the indices of all particles currently assigned to the Halo, meaning that are inside the potential well (and not necesarily bound to it)

        Outputs:
        ----------
        group: (Array of int) Array of the indices of all particles assigned to the Halo's potential well
        '''
        return self.group_tracker.get_particles()
    
    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_current_surface_particles(self):
        '''
        Returns 1D array of the indices of all particles currently connected/adjacent but not within the Halo's potential well

        Outputs:
        ----------
        surface: (Array of int) Array of the indices of all particles currently connected/adjacent but not within the Halo's potential well
        '''
        return self.surface_tracker.get_particles()

    cpdef list get_subgroups(self):
        '''
        Returns list of all subgroups within the main group. Each element contains a 1D array of the indices of all the particles assigned to that subgroup

        Outputs:
        ----------
        subgroups: (list of Arrays of int) list of arrays of indices of all particles assigned to subgroups
        '''
        return self.subgroups

    cpdef cnp.ndarray[long, ndim = 1, cast = True] get_bound_mask(self):
        '''
        Returns boolean mask array of the same size as get_current_group_particles selecting only particles that are dynamically bound to the Halo's potential well.

        Outputs:
        ----------
        bound_mask: (Array of bool) Mask selecting Halo particles dynamically bound to the potential well
        '''
        return self.to_bool_array(self.bound_mask)
    
cdef class ParticleAssigner:
    '''
    ParticleAssigner class. This class is designed to assign particles to a structure given a potential surface and local acceleration defining the boosted potential (for details see: https://arxiv.org/abs/2107.13008 and https://arxiv.org/abs/<INSERT NUMBER>). 
    
    The underlying algorithm relies on traveling an ensemble of connected particles in search of a saddle point in the boosted potential. It is designed to be agnostic of the particular geometry of the problem ans simply requires knowledge of the connectivity between nodes/particles. In general this means that the algorithm requires a list of the nearest neighbours to each particle to segement the different groups. 
    
    The main method of this class is 'segement' which selects all the connected particles conected to particle 'i0' given a local acceleration 'acc0'.
    
    Parameters:
    ----------
    ngbs: (list of arrays of int) list of neighbours to each particle, these take the form of arrays of indices pointing to the corresponding neighbours
    pot: (array of floats) global gravitation potential evaluated at the position of each particle
    pos: (array of floats) positions of particles in d-dimensional space.
    vel: (array of floats) velocities of particles in d-dimensional space.
    scale_factor: (float) Snapshot scale factor 'a' (defaults to a = 1.)
    Omega_m: (float) Density of dark matter in units of the critical density, note: flat cosmology is assumed (default to Einstein-de Sitter Omega_m = 1.)
    Lbox: (float) simulation boxsize (defaults to 1000 h^{-1}Mpc)
    H0: (float) Hubble parameter/constant (defaults to 100 km/s/Mpc)
    threshold: (str) indentifier for which value is used for the large scale correction delta parameter (Defaults to EdS-cond, delta = 0, see get_delta_th method for specifics).
    no_binding: (bool) switch turning off the binding check (defaults to False)
    verbose: (bool) switch toggling verbosity (defaults to False)
    custom_delta: (bool) switch which allows to ignore the threshold argument and provide a custom value of delta (defaults to False)
    delta: (float) custom value of delta, custom_delta must be True to use this option (defaults to 0)
    
    Methods:
    ----------
    [Cosmology, Assuming Flat LCDM]
    - update_cosmology: Update cosmological parameters and associated internal variables

    - H_a: Hubble parameter computed at scale factor a

    - H_dot: Time derivative of the Hubble parameter computed at scale factor a

    - D: Linear growth factor D, computed at scale factor a

    - g: D/a normalised such that g(a -> 0) = 1 

    - w: Omega_L(a)/Omega_m(a)

    - Omega_m: Omega_m(a)

    - Omega_L: Omega_Lambda(a)

    - time_of_a: Age of the universe at scale factor a

    - Newton_Raphson: Newton_Raphson root finder

    - get_zeta: LCDM spherical collapse zeta as defined in Mo et al. 2010

    - get_delta_th: Precomputed values of the large scale factor for several spherical collapse contexts
    
    [Particle Tracking]
    
    - phi_boost: returns the boosted potential at the postion of particle i (C only)

    - get_phi_boost:returns the boosted potential at the postion of an array particles
    
    - subfind_neighbours: (Untested) modifies the neighbour's list to correspond to a definition equivalent to the subfind halo finder. 
    
    - reciprocal_neighbours: (Untested) modifies the neighbour's list so that all neighbour connections are reciprocal, i.e all particles know of all particles it is a neighbour of.
    
    - fill_below: itterative watershed algorithm finding all conected particles with lower potentials segementing them into internal and surface sets which are saved internally.
    
    - fill_below_substructure: same as fill_below but explicitly takes the inner and surface sets as aguments instead of using internal variables.
    
    - grow: grows the sturcture itteratively by including one by one the particles with the lowest bosted potential until a saddle point is found.
    
    - segment: main user function, selects all the connected particles that are assigned to the structure.

    - is_bound:
    '''
    
    cdef Py_ssize_t nparts
    cdef Py_ssize_t nngbs
    
    cdef long[:,:] ngbs
    cdef cnp.double_t[:] pot
    cdef cnp.double_t[:,:] pos
    cdef cnp.double_t[:,:] vel
    
    cdef cbool verbose
    cdef cbool no_binding
    cdef str threshold

    cdef cnp.double_t Lbox
    cdef cnp.double_t _Omega_m
    cdef cnp.double_t _Omega_L
    cdef cnp.double_t _Omega_k
    cdef cnp.double_t _scale_factor
    cdef cnp.double_t _H0
    cdef cnp.double_t _Ha
    cdef cnp.double_t _Hd
    cdef cnp.double_t _delta_th
    cdef cnp.double_t _long_range_fac
    cdef cnp.double_t _zeta
    

        
    def __init__(self, cnp.ndarray[long, ndim = 2] ngbs, cnp.ndarray[double, ndim = 1] pot, cnp.ndarray[double, ndim = 2] pos, cnp.ndarray[double, ndim = 2] vel, 
                 cnp.double_t scale_factor = 1., cnp.double_t Omega_m = 1., cnp.double_t Lbox = 1000., cnp.double_t H0 = 100, str threshold = 'EdS-cond',
                 no_binding = False, verbose = False, cbool custom_delta = False,  cnp.double_t delta = 0.):
        
        self.nparts = ngbs.shape[0]
        self.nngbs = ngbs.shape[1]
        
        self.ngbs = ngbs
        self.pot = pot
        self.pos = pos
        self.vel = vel
        # self.ids_fof = ids_fof
        
        self.verbose = verbose
        self.no_binding = no_binding
        self.threshold = threshold
        
        self.Lbox = Lbox
        self._Omega_m = Omega_m
        self._Omega_L = 1. - Omega_m
        self._Omega_k = 0.
        self._scale_factor = scale_factor
        self._H0 = H0
        self._Ha = self.H_a(self._scale_factor)
        self._Hd = self.H_dot(self._scale_factor)
        if custom_delta:
            self._delta_th = delta
        else:
            self._delta_th = self.get_delta_th(threshold)

        self._long_range_fac =  0.25 * self._delta_th *self._Omega_m * self._scale_factor**-3 * self._H0 * self._H0
        if verbose:
            print(f"long range factor: {self._long_range_fac}")
        #
        return
    
    # ================== Cosmology ==============================
    # Everything should be in units of h
    
    def update_cosmology(self, scale_factor = None, Omega_m = None, Lbox = None):
        '''
        Updates internal cosmological parameters and recomputes internal properties.
        
        Parameters:
        ----------
        scale_factor: (float) Simulation scale factor.
        Omega_m: (float) Matter density parameter Omega_m.
        Lbox: (float) Simulation box size.
        '''
        if Lbox != None:
            self.Lbox = Lbox
        if Omega_m != None:
            self._Omega_m = Omega_m
            self._Omega_L = 1. - Omega_m
            self._Omega_k = 0.
        if scale_factor != None:
            self._scale_factor = scale_factor
            self._Ha = self.H_a(scale_factor)
            self._Hd = self.H_dot(scale_factor)
        self._delta_th = self.get_delta_th(self.threshold)
        self._long_range_fac = 0.25 * self._delta_th * self._Omega_m * self._scale_factor**-3 * self._H0 * self._H0
        return 
    
    def H_a(self, a):
        '''
        Computes the Hubble parameter H(a) at scale factor a
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        H_a: (float) Hubble parameter H(a)
        '''
        return self._H0 * np.sqrt(self._Omega_m/a**(3)+self._Omega_k/a**(2)+self._Omega_L)

    def H_dot(self, a):
        '''
        Computes the time derivative of Hubble parameter H(a) at scale factor a.
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        H_dot: (float) time derivative of the hubble parameter H(a)
        '''
        return -self._H0 / 2 / np.sqrt(self._Omega_m/a**(3)+self._Omega_k/a**(2)+self._Omega_L) * (3*self._Omega_m/a**(4) + 2*self._Omega_k/a**(3))
        
    def D(self, a):
        '''
        Computes the Linear growth factor D(a) at scale factor a
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        D: (float) Linear Growth factor D(a)
        '''
        f = lambda ap: (self._Omega_m*ap**(-1.) + self._Omega_L*ap**(2) + self._Omega_k)**(-3/2)
        d, err = self.H_a(a)/self._H0 * np.array(quad(f,1e-8,a))
        D0 = self.H_a(1.)/self._H0 * np.array(quad(f,1e-8,1.))[0]
        return d/D0

    def g(self, a):
        '''
        Computes the renormalise Linear growth factor g(a) at scale factor a. Normalised at a = 1e-5 such that g(1e-5) = 1. See Mo et al. (2010).
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        g: (float) Renomalised linear Growth factor D(a)/a
        '''
        g_i = self.D(1e-5)/1e-5
        return np.vectorize(self.D)(a)/a / g_i

    def w(self, a):
        '''
        Not dark energy equation of state. Computes the ratio of the dark energy and matter density parameters at scale factor a. See Mo et al. (2010).
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        D: (float) Linear Growth factor D(a)
        '''
        return (self._Omega_L/self._Omega_m) * a**3

    def Omega_m(self, a):
        '''
        Computes the matter density parameter at scale factor a
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        Om: (float) Matter density parameter Omega_m(a)
        '''
        return self._Omega_m * a**-3 * self._H0*self._H0/self.H_a(a)/self.H_a(a) 

    def Omega_L(self, a):
        '''
        Computes the dark energy density parameter at scale factor a
        
        Parameters:
        ----------

        a: (float) Scale factor
        
        Outputs:
        ----------
        OL: (float) Dark energy density parameter Omega_Lambda(a)
        '''
        return self._Omega_L * self._H0**2/self.H_a(a)**2 

    #def t_H(self):
    #    return 9.7779222 #h^{-1}Gyr

    def time_of_a(self, a):
        '''
        Computes the age of the Universe at scale factor a in units of the Hubble time.
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        T: (float) Age of the Universe at scale factor a
        '''
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
        This is the standard Newton-Raphson root finder "borrowed" from Numerical Recipes 3rd edition (Press et al. 2003)
        
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
        '''
        Mo et al. 2010 zeta variable used to compute the spherical turn around and collapse overdensity in Lambda CDM. See Chap. 5 for details. Here we compute this variable numerically rather than using the analytical approximation given by Mo et al.
        
        Parameters:
        ----------
        a: (float) Scale factor
        
        Outputs:
        ----------
        zeta: (float) value of zeta at scale factor a
        '''
        def func(zeta, a):
            t_max = self.time_of_a(a)
            dy = lambda x: 1/np.sqrt(1/x - 1 + zeta * (x*x -1))
            y,err = quad(dy, 0, 1)
            return  t_max - np.sqrt(zeta/self._Omega_L) * y

        zeta = self.Newton_Raphson(f = lambda z: func(z,a), xi = 0.1, dx = 1e-11, tol = 1e-10)
        return zeta
    
    def get_delta_th(self, threshold):
        '''
        Function computing various theoretically motivated values for the large scale correction parameter delta. These are accessed using an identifier string. For formulae and specifics see Mo et al. 2010
        Recognized options: 
        "EdS-cond" Condition for collapse in EdS cosmology (delta = 0), 
        "EdS-coll" Linearly extrapolated overdensity at collapse in EdS cosmology, 
        "EdS-ta-lin" Linearly extrapolated overdensity at turn-around in EdS cosmology, 
        "EdS-ta-eul" Eularian overdensity at turn-around in EdS cosmology, 
        "LCDM-cond" Condition for collapse in LCDM cosmologies, 
        "LCDM-coll" Linearly extrapolated overdensity at collapse in LCDM cosmologies, 
        "LCDM-ta-lin" Linearly extrapolated overdensity at turn-around in LCDM cosmologies, 
        "LCDM-ta-lin" Eularian overdensity at turn-around in LCDM cosmologies.
        
        Parameters:
        ----------
        threshold: (str) model identifier
        
        Outputs:
        ----------
        delta: (float) value of delta for the given model and computed using the internal cosmological parameters.
        '''
        
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
                raise ValueError(f'threshold definition {threshold} was not a recognized option:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
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
                raise ValueError(f'threshold definition {threshold} was not a recognized option:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
        else:
            raise ValueError(f'threshold definition {threshold} was not a recognized option:\n "EdS-cond", "EdS-coll", "EdS-ta-lin", "EdS-ta-eul", "LCDM-cond", "LCDM-coll", "LCDM-ta-lin", and "LCDM-ta-lin"')
        return res
    
    

    
    # ================== Utility Methods ========================
    
    def to_bool_array(self, lst):
        '''
        Transforms cython uint8 MemoryView into numpy boolean array that is usable as a mask

        Parameters:
        ----------
        lst: 1-D MemoryView of dtype uint8

        Outputs:
        ----------
        res: 1-D NumPy array of dtype bool
        '''
        cdef cnp.ndarray[cnp.uint8_t, ndim = 1, cast=True] res
        res = np.array(lst, dtype=bool)
        return res

    cpdef cnp.double_t get_long_range_fac(self):
        '''
        Returns potential long range correction factor in physical units 0.25 * delta * Omega_m * a^-3 * H_0^2
        Outputs:
        ----------
        long_range_fac (float) potential long range correction factor
        '''
        return self._long_range_fac
    
    #####=================== Boost!!! ===========================
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
    
    cdef cnp.double_t phi_boost(self, long i, Halo halo):
        '''
        Function which calculates the boosted potential at the position of particle i
        
        Parameters:
        ----------
        i: (int or array of ints) index/array of indices of the particles for which to calculate the boosted potential.
        halo: (Halo) halo with respect to which the boosted potential is to be computed
        
        Returns:
        ----------
        phi_boost: (float or array of floats) boosted potential of the requested particles
        '''
        
        if halo.x0 is None:
            raise ValueError('A reference position x0 must be set first. This can be done by calling the set_x0 or segment methods.')
        if halo.acc0 is None:
            raise ValueError('A reference acceleration acc0 must be set first. This can be done by calling the set_acc0 or segment methods.')
            
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx, temp_xa
        cdef int k
        cdef cnp.double_t res
        cdef cpair[cnp.double_t, long] elem
        
        if halo.computed_tracker.is_mask(i):
             res = halo._phi[i]
        else:
            x = self.recentre_positions(self.pos[i,:], halo.x0)
            
            temp_xx = 0.0
            temp_xa = 0.0
            for k in range(len(x)):
                temp_xx += x[k]*x[k]
                temp_xa += x[k]*halo.acc0[k]
            res = (self.pot[i] - self.pot[halo.i0]) / self._scale_factor \
                            + temp_xa / self._scale_factor \
                            - self._long_range_fac * temp_xx  * self._scale_factor * self._scale_factor
            
            # We mark this particle as already computed.
            elem = (res, i)
            halo.computed_tracker.add_elem(elem)
            halo._phi[i] = res

        return res
    
    def get_phi_boost(self, indices, halo):
        '''
        Function which calculates the boosted potential at the position of particle i
        
        Parameters:
        ----------
        i: (int or array of ints) index/array of indices of the particles for which to calculate the boosted potential.
        halo: (Halo) halo with respect to which the boosted potential is to be computed
        
        Returns:
        ----------
        phi_boost: (float or array of floats) boosted potential of the requested particles
        '''
        
        indices = np.array(indices)
        if indices.size == 1:
            indices = np.array([indices.item(),])
        res = np.zeros(indices.size, dtype = 'f8')

        
        for j,i in enumerate(indices):
            res[j] = self.phi_boost(i, halo)
        if res.size == 1:
            return res.item()
        return res
    
    #####=================== Neighbours ================== 
    ## These are untested, deprecated, and will be deleted
    
    def subfind_neighbours(self, Halo halo):
        '''
        Function altering the list of neighbours to obtain a connectivity equivalent to that used within the subfind algorithm. See Springel et al. 2001, MNRAS 328, 726–750 for details on the subfind algorithm. Warning this functgion will rewrite the entire neighbours list and as such can take an extremely long amount of time if used on too many particles.
        
        Parameters:
        ----------
        halo: (Halo) Halo for which the boosted potenital will be computed
        '''
        cdef int i, j
        recip_ngbs = list(self.ngbs)
        
        for i in range(self.pos.shape[0]):
            cond = self.phi_boost(recip_ngbs[i], halo) < self.phi_boost(i, halo)
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
        '''
        cdef int i,j
        cdef list recip_ngbs = list(self.ngbs)
        
        for i in range(self.pos.shape[0]):
            for j in recip_ngbs[i]:
                if i not in set(recip_ngbs[j]):
                    recip_ngbs[j] = np.array(list(recip_ngbs[j]) + [i,])
        self.ngbs = recip_ngbs
        return
    
    # ================== Particle Assignment ====================
    
    cpdef long first_minimum(self, long i0, Halo halo, double r = 1):
        '''
        Function finding a minimum in the potential starting as defined for "halo". The search proceeds as an approximate gradient decent following the connectivity until a particle with no neighbours with lower potential values is found. 
        
        Parameters:
        ----------
        i0: (int) index of first guess particle
        halo: (Halo) halo with respect to which the boosted potential is computed

        Outputs:
        ----------
        i: (int) index of the minimum
        '''
        
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
            phi_ngbs[j] = self.phi_boost(ngbs_temp[j], halo)

        while np.min(phi_ngbs) < self.phi_boost(i, halo):

            counter += 1
            i = ngbs_temp[np.argmin(phi_ngbs)]
            ngbs_temp = self.ngbs[i]
            phi_ngbs = np.zeros(len(ngbs_temp))
            for j in range(len(ngbs_temp)):
                phi_ngbs[j] = self.phi_boost(ngbs_temp[j], halo)

                
            if counter % 100 == 0:
                x = self.recentre_positions(self.pos[i], self.pos[i0])
                temp_xx = 0.0
                for k in range(len(x)):
                    temp_xx += x[k]*x[k]
                dist = np.sqrt(temp_xx)
                if dist > r:
                    if self.verbose: print(f"The minimum is {float(dist):.3} (> {float(r):.3}) Mpc away from the starting position.\
                                         \nThis may indicate that there is either no miminum or that you are starting index is too far away.", flush = True)
                    halo._too_far = True
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
            halo._too_far = True
            return -1
        return i
    
    cpdef long fof_minimum(self, long[:] ids_fof, Halo halo):
        '''
        First minimum search using a prior group of particle indices (ids_fof) and computing the boosted potential with respect to halo. Computes the boostded potential for particles in ids_fof and returns the index of the particle with the lowest boosted potential.
        
        Parameters:
        ----------
        ids_fof: (Array of int) array of the indices of particles in the prior group
        halo: (Halo) halo withrespect to which to compute the boosted potential

        Outputs:
        ----------
        i: (int) index of the particle in ids_fof with the lowest boosted potential
        '''
        cdef long k, i_min
        cdef cnp.double_t phi_min
        cdef cpair[cnp.double_t, long] elem
        cdef cpp_pq loc_queue = cpp_pq(compare_first)
        for k in ids_fof:
            elem = (self.phi_boost(k, halo), k)
            
            loc_queue.push(elem)
        elem = loc_queue.top()
        return elem.second
    
    cpdef long itt_minimum(self, long i0, Halo halo, cnp.double_t r = 1.):
        '''
        Function that combines first_minimum and fof_minimum to find the minimum of the potential. If "halo "has an prior group (ids_fof), i0 is ignored and fof_minimum is used to start the approximate gradient descent.
        
        
        Parameters:
        ----------
        i0: (int) index of starting particle, ignored if halo has a prior group (halo.ids_fof)
        halo: (Halo) halo with respect to which the boosted potential is computed.
        r: (float) maximum travel distance, raises ValueError if the gradient descent stray further than r from the starting particle.

        Outputs:
        ----------
        i: (int) index of the particle with the lowest boosted potential.
        '''
        cdef long i0_itt = i0
        cdef long i0_fof, i0_temp
        cdef cbool min_found = False
        cdef set mem = set()

        if len(halo.ids_fof) > 0:
            i0_fof = self.fof_minimum(halo.ids_fof, halo)
            i0_fof = self.first_minimum(i0_fof, halo, r)
            if i0_fof not in set(halo.ids_fof) or halo._too_far:
                if self.verbose: print(f"The initial minimum is outside of the seed group falling back to standard approach.", flush = True)
                if halo._too_far: halo._too_far = False
                i0_temp = self.first_minimum(i0_itt, halo, r)
                if i0_temp not in set(halo.ids_fof) or halo._too_far:
                    if self.verbose: print(f"The initial minimum is still outside of the seed group. Exiting", flush = True)
                    return -1
            else:
                i0_itt = i0_fof
        else:
            i0_itt = self.first_minimum(i0, halo, r)
        
        halo.set_x0(i0_itt, self.pos[i0_itt])
        
        if i0_itt == self.first_minimum(i0_itt, halo, r):
            min_found = True
            i0 = i0_itt
        
        while not min_found: # Itterate over first minimum to check that it is indeed a minimum in it's own reference frame
            i0_temp = self.first_minimum(i0_itt, halo, r)
            if i0_temp == -1 or halo._too_far:
                if self.verbose: print(f"Something went wrong itterating over minima. Exiting", flush = True)
                return -1
            else: i0_itt = i0_temp
            if len(halo.ids_fof) > 0: # If we have a fof seed make sure we don't leave the group through here.
                if i0_itt not in set(halo.ids_fof) or halo._too_far:
                    if self.verbose: print(f"The initial minimum is still outside of the seed group. Exiting", flush = True)
                    return -1

            halo.set_x0(i0_itt, self.pos[i0_itt])
            if i0_itt == self.first_minimum(i0_itt, halo, r): # This particle is the minimum in it's own refference frame.
                min_found = True
                i0 = i0_itt

            if i0_itt in mem: # If we end up here we're in a multiple particle loop
                if self.verbose: print(f"Stuck in a loop finding first minimum.\n{mem}\n Exiting", flush = True)
                return -1
            mem.add(i0_itt) # Keep track of which particles have been visited to avoid getting stuck in a loop
        return i0
    
    cpdef void fill_below(self, cnp.int64_t i, Halo halo):
        '''
        Itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle i. 
        
        Note that the particles are saved within a set 'self.i_in' that is saved internally to the class and all the direct neighbours to the particles in this set are saved to a second set 'self.i_surf'. 
        
        These sets should not be modified directly as this can cause issues with the segmentation algorithm.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        halo: (Halo) halo with respect to which the boosted potential is to be computed, and within which data will be stored
        '''
        
        cdef double phi_curr = self.phi_boost(i, halo)
        cdef long[:] ngbs_temp
        cdef long j, k, l, index
        #cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.ndarray[long, ndim = 1, cast = True] indices
        cdef cpair[cnp.double_t, long] elem, top_elem
        
        # Reset Trackers
        halo.group_tracker.reset()
        halo.surface_tracker.reset()
        
        # Put starting particle in group
        #self.group[i] = self._current_group
        halo.visited[i] = True
        elem = (self.phi_boost(i, halo), i)
        halo.group_tracker.add_elem(elem)
        
        # Calculate it's potential and define its neighbours as surface particles
        
        ngbs_temp = self.ngbs[i]
        for k in ngbs_temp:
            if not halo.visited[k]:
                elem = (self.phi_boost(k, halo), k)
                halo.surface_tracker.add_elem(elem)
        
        # Take surface particle with lowest potential
        
        elem = halo.surface_tracker.get_top_elem()
        halo.surface_tracker.remove_top_elem()
        j = elem.second
        
        while self.phi_boost(j, halo) - phi_curr < 0.0:

            halo.visited[j] = True
            elem = (self.phi_boost(j, halo), j)
            halo.group_tracker.add_elem(elem)
            
            ngbs_temp = self.ngbs[j]
            
            for k in ngbs_temp:
                if halo.surface_tracker.is_mask(k) or halo.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    elem = (self.phi_boost(k,halo), k)
                    halo.surface_tracker.add_elem(elem)

            top_elem = halo.surface_tracker.get_top_elem()
            halo.surface_tracker.remove_top_elem()
            j = top_elem.second
        return
    

    
    cdef void fill_below_substructure(self, long i, Halo halo, double phi_min):
        '''
        itterative watershed function which finds all the particles continuosly connected to particle i that have a boosted potential lower than particle k. 
        
        Note that this function is similar to check(i,k), but has the major difference of explicitly taking the sets of indices 'i_in' and 'i_surf' as arguments. 
        This allows to probe newly found potential wells without altering the sets of particles currently assigned to the man structure or its surface.
        
        Parameters:
        ----------
        i: (int) index of the particle defining the level of the potential
        halo: (Halo) halo with respect to which the boosted potential is to be computed, and within which data will be stored
        phi_min: (float) current minimum of the potential well
        '''
        #cdef cnp.ndarray[long, ndim = 1, cast = True] id_surf
        cdef cnp.double_t[:] x
        cdef cnp.double_t temp_xx
        cdef double dist
        cdef double phi_curr = self.phi_boost(i, halo)
        cdef long j, k, index, j_old
        cdef long[:] ngbs_temp
        cdef int counter = 1
        cdef double phi_j
        
        # Empty queue
        cdef cpair[cnp.double_t, long] elem

        if not halo.subsurface_tracker.is_empty():
            halo.subsurface_tracker.reset()
        if not halo.subgroup_tracker.is_empty():
            halo.subgroup_tracker.reset()
        
        # Start by adding the refference particle to the group
        
        halo.visited[i] = True
        elem = (self.phi_boost(i,halo),i)
        halo.subgroup_tracker.add_elem(elem)

        # Calculate it's potential and define its neighbours as surface particles
        
        ngbs_temp = self.ngbs[i]

        
        for k in ngbs_temp:
            if not halo.visited[k]:
                elem = (self.phi_boost(k, halo), k)
                halo.subsurface_tracker.add_elem(elem)
        
        if halo.subsurface_tracker.is_empty():
            if self.verbose: 
                print(f"subsurface queue started empty {i}", flush = True)
                print(f"ngbs: {[ngb for ngb in self.ngbs[i]]}", flush = True)
                print(f"pots: {[self.phi_boost(ngb,halo) for ngb in self.ngbs[i]]}", flush = True)
                print(f"visited: {[halo.visited[ngb] for ngb in self.ngbs[i]]}", flush = True)
                #print(f"groups: {[self.group[ngb] for ngb in self.ngbs[i]]}", flush = True)
            return
        
        elem = halo.subsurface_tracker.get_top_elem()
        j = elem.second
        j_old = -1
        
        while self.phi_boost(j, halo) - phi_curr < 0:
            #========= Test ================ 
            if halo.visited[j]: # If we've already put j in the subgroup or is part of the main group
                if self.verbose: print(f"Particle {j} already visited but in subsurface queue...", flush = True)
                halo.subsurface_tracker.remove_top_elem()
                
                if halo.subsurface_tracker.is_empty():
                    #self._too_far = True
                    if self.verbose: print(f"subsurface queue empty {j}", flush = True)
                    return
                elem = halo.subsurface_tracker.get_top_elem()
                j = elem.second
                continue
            
            
            if j == j_old:
                halo._too_far = True
                if self.verbose: print(f"ya in a loop it seems {j}", flush = True)
                return
            j_old = j
            #=============================== 
            counter += 1        
            # By taking this one we ensure we are always going down the lowest brach first.
            phi_j = self.phi_boost(j, halo)
            
            if counter % 100 == 0: # Check if we haven't wandered too far away
                x = self.recentre_positions(self.pos[j], halo.x0)
                temp_xx = 0.0
                for k in range(len(x)):
                    temp_xx += x[k]*x[k]
                dist = np.sqrt(temp_xx)
                if dist > 3*halo.max_dist:
                    if self.verbose:
                        print(f"The new minimum is {float(dist):.3} (> {float(3*halo.max_dist):.3}) Mpc away from the starting position.\nThis may indicate that there is either no miminum or that you are starting index is too far away.")
                    halo._too_far = True
                    return
    
                if halo.subgroup_tracker.get_size() > halo.group_tracker.get_size():
                    if self.verbose:
                        print(f"Trying to add in a structure which is much larger than the current structure {halo.subgroup_tracker.get_size()} > {halo.group_tracker.get_size()}. Exiting.")
                    halo._too_far = True
                    return
            
            # If we are in this we already know that j has a lower potential
            # So we can add it directly and remove it from the surface

            halo.subsurface_tracker.remove_top_elem()
            halo.visited[j] = True
            halo.subgroup_tracker.add_elem(elem)
            
            if phi_j < phi_min: # Check to see if we are going below the current minimum potential
                halo.new_min = phi_j
                return
            # Add neighbours to surface
            for k in self.ngbs[j]:
                if halo.subsurface_tracker.is_mask(k) or halo.visited[k]:
                    # Avoid duplicates or going back to a particle that has already been visited
                    continue
                else: 
                    # Insert particle in sorted order
                    elem = (self.phi_boost(k,halo), k)
                    halo.subsurface_tracker.add_elem(elem)
            
            #========= Test ================        
            if halo.subsurface_tracker.is_empty():
                #self._too_far = True
                if self.verbose: print(f"subsurface queue empty {j}", flush = True)
                return
            #=============================== 
            # Take subsurface particle with lowest potential
            elem = halo.subsurface_tracker.get_top_elem()
            j = elem.second
        return
    
    cpdef tuple grow(self, Halo halo):
        '''
        Itterative algorithm which grows the structure one particle at a time. At each step the particle with the lowest potential on the surface is considered. If all of its neighbours with lower potential currently belongs to the structure then it is also added to the structure. If any of its lower neighbours have lower potentials and are NOT associated to the structure then the particle is marked as a saddle point. These braches are then explored with a watershed algorithm to find if any brach has reaches values that are deeper than the current minimum of the boosted potential. If so the algorithm terminates.
        
        Parameters:
        ----------
        halo: (Halo) halo with respect to which the boosted potential is to be computed, and within which data will be stored
        
        Returns:
        ----------
        i_min: (int) index of the particle with the lowest boosted potential in the group
        i_sad: (int) index of the particle corresponding to the saddle point in the boosted potetnial
        '''
        cdef double phi_min, phi_max
        
        cdef long i_prev, n_part
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
        cdef list subgroup
        
        group_particles = halo.get_current_group_particles()
        n_part = group_particles.size
        phi_min = self.phi_boost(group_particles[0], halo)
        i_min = group_particles[0]
        phi_max = self.phi_boost(group_particles[0], halo)
        i_max = group_particles[0]
        
        for k in group_particles:
            x = self.recentre_positions(self.pos[k], halo.x0)
            temp_xx = 0.0
            for k_loc in range(len(x)):
                temp_xx += x[k_loc]*x[k_loc] 
            dist = np.sqrt(temp_xx)
            if dist > halo.max_dist:
                halo.max_dist = dist
            phi_k = self.phi_boost(k, halo)
            if phi_k < phi_min:
                phi_min = phi_k
                i_min = k
            if phi_k > phi_max:
                phi_max = phi_k
                i_max = k
                
        i_prev = -1

        for _ in range(len(self.pot)): # <== I just don't want to write 'while True:'
            
            if halo.surface_tracker.is_empty():
                if self.verbose: print('Surface empty', flush = True)
                break
            
            qelem = halo.surface_tracker.get_top_elem()
            phi_cons = qelem.first
            i_cons = qelem.second
            
            if i_cons == i_prev:
                raise RecursionError(f"The algorithm seems to have gotten stuck at i = {i_cons} with {halo.group_tracker.get_size()} particles assigned and {halo.surface_tracker.get_size()} pending...")
            i_prev = i_cons
            
            if halo.visited[i_cons]: # If we've already put i_cons in the group but it was still in the surface.
                if self.verbose: print(f"{i_cons} had already been visited but was still in surface queue")
                halo.subsurface_tracker.remove_top_elem()
                continue
                
            halo.visited[i_cons] = True
            ngbs_temp = self.ngbs[i_cons]
            
            low_pot_cond = False
            for j in range(len(ngbs_temp)):
                if self.phi_boost(ngbs_temp[j], halo) < self.phi_boost(i_cons, halo):
                    low_pot_cond = True
                    break

            x = self.recentre_positions(self.pos[i_cons], halo.x0)
            temp_xx = 0.0
            for k_loc in range(len(x)):
                temp_xx += x[k_loc]*x[k_loc] 
            dist = np.sqrt(temp_xx)
            if dist > halo.max_dist:
                halo.max_dist = dist
            
            if not low_pot_cond:
                # This can happen when hitting boundaries, or for very small local minima
                # These particles are recaptured by the code later if we just skip them.
                if self.verbose: 
                    print(f"{i_cons} low_pot_cond failed: Has no neighbours with lower potential", flush = True)
                    print(f"Relative potentials: {[self.phi_boost(ngb, halo) - self.phi_boost(i_cons, halo) for ngb in self.ngbs[i_cons]]}", flush = True)
                    #print(f"Neighbouring groups: {[self.group[ngb] for ngb in self.ngbs[i_cons]]}", flush = True)
                    print(f"Neighbouring visits: {[halo.visited[ngb] for ngb in self.ngbs[i_cons]]}", flush = True)
                    print(f"Neighbouring surface: {[halo.surface_tracker.is_mask(ngb) for ngb in self.ngbs[i_cons]]}", flush = True)
            
                halo.surface_tracker.remove_top_elem()
                halo.visited[i_cons] = False
                continue
            
            if_cond = True
            
            for k, elem in enumerate(ngbs_temp):
                if self.phi_boost(elem, halo) < self.phi_boost(i_cons, halo):
                    if_cond *= halo.group_tracker.is_mask(elem)
            
            if if_cond:
                # Tag i_cons as part of the main group
                qelem = (phi_cons, i_cons)
                #self.group[i_cons] = self._current_group
                halo.group_tracker.add_elem(qelem)
                
                # Remove it from the surface
                halo.surface_tracker.remove_top_elem()
                
                if phi_cons < phi_min:
                    phi_min = phi_cons
                    i_min = i_cons
                if phi_cons > phi_max:
                    phi_max = phi_cons
                    i_max = i_cons

                for k in ngbs_temp:
                    if halo.surface_tracker.is_mask(k) or halo.visited[k]:
                        # Avoid duplicates or going back to a particle that has already been visited
                        continue
                    else:
                        # Insert particle in sorted order
                        qelem = (self.phi_boost(k, halo), k)
                        halo.surface_tracker.add_elem(qelem)

            else:
                # Or check if there is a substructure
                halo.visited[i_cons] = False
                halo.new_min = phi_cons
                
                halo.subsurface_tracker.reset()
                halo.subgroup_tracker.reset()
                
                self.fill_below_substructure(i_cons, halo, phi_min)
                
                if halo._too_far:
                    # Traveled too far or adding in a large group.
                    if self.verbose: print('Traveled too far exitting... ', halo.subgroup_tracker.get_size(), end = ' ', flush = True)
                    # Here we do a manual reset as we need to also reset the 'visited' array
                    while not halo.subgroup_tracker.is_empty():
                        qelem = halo.subgroup_tracker.get_top_elem()
                        halo.subgroup_tracker.remove_top_elem()
                        k = qelem.second
                        halo.visited[k] = False
                    
                    while not halo.subsurface_tracker.is_empty():
                        qelem = halo.subsurface_tracker.get_top_elem()
                        halo.subsurface_tracker.remove_top_elem()
                        k = qelem.second
                        halo.visited[k] = False
            
                    halo._too_far = False
                    break
                
                if halo.new_min <= phi_min:
                    # Moved into a lower potential well => exit
                    if self.verbose: print('Found lower minimum:', halo.subgroup_tracker.get_size(), i_cons, end = ' ', flush = True)
                    # Make saddle point a part of the main group
                    # Tag i_cons as part of the main group
                    
                    #self.group[i_cons] = self._current_group
                    qelem = (phi_cons, i_cons)
                    halo.group_tracker.add_elem(qelem)
                    
                    # Remove it from the surface
                    halo.surface_tracker.remove_top_elem()
                    
                    if phi_cons < phi_min:
                        phi_min = phi_cons
                        i_min = i_cons
                    if phi_cons > phi_max:
                        phi_max = phi_cons
                        i_max = i_cons
                        
                    # For completeness add it's neighbours to the surface.
                    for k in ngbs_temp:
                        if halo.surface_tracker.is_mask(k) or halo.visited[k]:
                            # Avoid duplicates or going back to a particle that has already been visited
                            continue
                        else:
                        
                            # Insert particle in sorted order
                            qelem = (self.phi_boost(k,halo), k)
                            halo.surface_tracker.add_elem(qelem)
                    
                    # Here we do a manual reset as we need to also reset the visited array
                    while not halo.subgroup_tracker.is_empty():
                        qelem = halo.subgroup_tracker.get_top_elem()
                        halo.subgroup_tracker.remove_top_elem()
                        k = qelem.second
                        halo.visited[k] = False
                    
                    while not halo.subsurface_tracker.is_empty():
                        qelem = halo.subsurface_tracker.get_top_elem()
                        halo.subsurface_tracker.remove_top_elem()
                        k = qelem.second
                        halo.visited[k] = False
                    
                    break
                    
                
                else:
                    # Found a structure => add it in 
                    # Merge surfaces
                    halo.visited[i_cons] = True 
                    # Remove i_cons from sufrace
                    halo.surface_tracker.remove_top_elem()
                    # Merge subsurface and surface
                    while not halo.subsurface_tracker.is_empty():
                        # remove lowest potential subsurface particle
                        qelem = halo.subsurface_tracker.get_top_elem()
                        k = qelem.second
                        halo.subsurface_tracker.remove_top_elem()
                        
                        if halo.surface_tracker.is_mask(k) or halo.visited[k]:
                            # Avoid duplicates or going back to a particle that has already been visited
                            continue
                        else:
                            # Add it to surface tracker
                            halo.surface_tracker.add_elem(qelem)
                    
                    # Merge Group and Subgroup
                    # If subgroup large enough tag particles for follow up
                    size_cond = halo.subgroup_tracker.get_size() > 20
                    if size_cond:
                        subgroup = []
                        halo._current_subgroup += 1
                    while not halo.subgroup_tracker.is_empty():
                        qelem = halo.subgroup_tracker.get_top_elem()
                        halo.subgroup_tracker.remove_top_elem()
                        k = qelem.second
                        phi_k = qelem.first
                        # By construction there can't be any duplicates to to save time we skip checking and merge directly...
                        # If the code is modified check that you don't introduce duplicates because they break the priority queues
                        halo.visited[k] = True
                        #self.group[k] = self._current_group
                        halo.group_tracker.add_elem(qelem)
                        
                        if size_cond:
                            subgroup.append(k)
                            #self.subgroup[k] = self._current_subgroup
                            
                        x = self.recentre_positions(self.pos[k], halo.x0)
                        temp_xx = 0.0
                        for k_loc in range(len(x)):
                            temp_xx += x[k_loc]*x[k_loc] 
                        dist = np.sqrt(temp_xx)
                        if dist > halo.max_dist:
                            halo.max_dist = dist
                        
                        # Keep min and max up to date
                        if phi_k < phi_min:
                            phi_min = phi_k
                            i_min = k
                        if phi_k > phi_max:
                            phi_max = phi_k
                            i_max = k      
                    if size_cond:
                        halo.subgroups.append(subgroup)
                        
                if halo.group_tracker.get_size() > 0.25*len(self.pot):
                    # Temporary measure to get a catalogue in EdS-cond, This is a last resort and in practice should never happen
                    if self.verbose: print('Reached size limit (npart/4):', halo.group_tracker.get_size(), end = ' ', flush = True)
                    break

        halo.i_min = i_min
        halo.i_max = i_max
        return i_min, i_max
    
    cpdef is_bound(self, long i_min, long i_sad, Halo halo):
        '''
        Binding check. compares the mechanical energy Kinetic + boosted Potential of all particles in i_in to the boosted potential evajulated at the location of i_sad, representing a saddle point.
        
        Parameters:
        ----------
        i_in: (set of N int) indices of connected particles with boosted potential below the saddle point energy.
        i_sad: (int) particle id of the saddle point particle
        halo: (Halo) halo with respect to which the boosted potential is to be computed, and within which data will be stored
        
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
        
        i_in_arr = halo.get_current_group_particles()
        
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
            v_mean[j] = np.mean(v_in[:,j]) # Calculate the mean velocity of the group
        

        x = self.recentre_positions(self.pos[i_sad],self.pos[i_min])
        
        temp_xx = 0.0
        
        for j in range(len(x)):
            temp_xx += x[j]*x[j]
            
        factor_xx_K = 0.5*self._Ha**2 * self._scale_factor*self._scale_factor
        factor_xx_phi = 0.25*self._H0**2 * (self._Omega_m / self._scale_factor - 2 * self._Omega_L * self._scale_factor*self._scale_factor)
        factor_xv = self._scale_factor * self._Ha
        
        phi_p_sad = self.phi_boost(i_sad, halo) + factor_xx_phi * temp_xx
        phi_p_min = self.phi_boost(i_min, halo) # By construction is 0 but we write it anyway, just in case
        
        sqrt_a = np.sqrt(self._scale_factor)
        
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
            
            
            K[i] = 0.5 * temp_vv + factor_xv * temp_xv + factor_xx_K * temp_xx # <= Explicit expansion terms
            phi_p[i] = self.phi_boost(index, halo) + factor_xx_phi * temp_xx + self._long_range_fac * temp_xx * self._scale_factor * self._scale_factor # <= Converted to physical potential
            
            E[i] = K[i] + phi_p[i] - phi_p_min 
            if E[i] < phi_p_sad - phi_p_min:
                bound_mask[i] = True
        
        return bound_mask

    
    cpdef segment(self, long i0, cnp.double_t[:] acc0, cnp.ndarray[long, ndim = 1] ids_fof, cnp.double_t r = 1., Halo reuse_halo = None):
        '''
        Main user function. Segments all particles belong to the same group as particle 'i0'. The boosted potential here is calculated with respect to the position of particle 'i0' and with the acceleration vector 'acc0'. Prior initial guess particles need to be provide using ids_fof, 
        
        Parameters:
        ----------
        i0: (int) index of starting particle
        acc0: (array of float) acceleration vector used to calculate the boosted potential
        ids_fof: (array of int) particle ids of prior group to be used as an initial seed
        r: (float) initial maximum distance initial potential search can travel, is adaptively expanded during growth of the seed 
        reuse_halo: (Halo) pre-initialised halo within which data will be stored, allows to save significant allocation overhead although ensure that the halo has been reset
        
        Returns:
        ----------
        i_in: (set of ints) set of indices of all the particles contained in the structure
        i_min: (int) index of the particle with the lowest boosted potential in the group
        i_sad: (int) index of the particle corresponding to the saddle point in the boosted potetnial
        halo: (Halo) halo with respect to which the boosted potential is to be computed, and within which data will be stored
        
        '''
        cdef long i_min, i_sad, i0_fof, i0_temp, i0_itt
        cdef set mem = set()
        cdef cbool min_found
        cdef Halo halo
        if reuse_halo is None: # If we aren't reusing a Halo, create a new one
            halo = Halo(self.nparts, ids_fof)
        else: # If we are, clean it
            halo = reuse_halo
            halo.reset_computed_particles()
        halo.set_acc0(acc0)
        halo.set_x0(i0, self.pos[i0])
        
        halo._current_subgroup = 0 # This simply resets the subgroup index which is currently unused
        
        # Verify that the first minimum is not too far away.
        
        i0 = self.itt_minimum(i0, halo, r)
        
        if i0 == -1:
            return np.array([], dtype = 'i8'), i0, i0, halo
        
        # Find all particles with potential lower than minimum (This should only give back 1 particle)
        halo.set_x0(i0, self.pos[i0])
    
        self.fill_below(i0, halo)
        
        # Grow potential surface
        i_min, i_sad = self.grow(halo)
    
        # Binding check
        if self.no_binding: 
            return halo.get_current_group_particles(), i_min, i_sad, halo
        bound_mask = self.is_bound(i_min, i_sad, halo)
        halo.bound_mask = bound_mask
        return halo.get_current_group_particles()[bound_mask], i_min, i_sad, halo # output: i_min, i_sad, n_part(, subs, i_in)
    
    