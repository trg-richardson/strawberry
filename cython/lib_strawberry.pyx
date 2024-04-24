#cython: language_level=3
#import numpy as np
cimport numpy as np


# Declare the function with types for better performance
cpdef np.ndarray[double] double_elements_cython(np.ndarray[double] arr):
    # Create a new array to store the results
    cdef np.ndarray[double] result = np.empty_like(arr)
    
    # Loop through each element of the input array and double its value
    cdef int i
    for i in range(len(arr)):
        result[i] = arr[i] * 2
        
    return result

def is_sorted(a):
    return np.all(np.diff(a) >= 0)

cpdef long binary_search(element, l):
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
    if not is_sorted(l): raise ValueError("The list provided is not sorted in increasing order.")
    cdef np.ndarray[long] arg = np.arange(len(l))
    cdef Py_ssize_t N
    
    while len(l) > 1:
        N = len(l)
        if l[N//2] > element:
            l = l[:N//2]
            arg = arg[:N//2]
        else:
            l = l[N//2:]
            arg = arg[N//2:]
    arg = arg + 1
    if arg == 1 and element < l:
        arg = np.zeros(1, 'i8')
    return arg.item()

cpdef np.ndarray[long] unmix(np.ndarray[long] a, np.ndarray[long] order):
    cdef np.ndarray[long] a_new = np.zeros(len(a), dtype = long)
    for i,j in enumerate(order):
        a_new[j] = a[i]
    return a_new
