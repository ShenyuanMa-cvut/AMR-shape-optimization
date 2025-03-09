# distutils: language=c++
# cython: language_level=3

from libcpp.set cimport set

import cython

import numpy as np
cimport numpy as np

ctypedef np.int32_t NP_INT
ctypedef np.float64_t NP_FLOAT

cdef class _AdjHelper():
    cdef readonly int[:] _array
    cdef readonly int[:] _offsets

    def __init__(self, np.ndarray[NP_INT,ndim=1] array, np.ndarray[NP_INT,ndim=1] offsets):
        self._array = np.array(array)
        self._offsets = np.array(offsets)

    cpdef int[:] links(self, int i):
        return self._array[self._offsets[i]:self._offsets[i+1]]

cdef class _MeshHelper():
    cdef readonly int dim
    cdef readonly int ncells
    cdef readonly NP_FLOAT[:,:] x
    cdef readonly NP_FLOAT[:] h
    cdef readonly _AdjHelper c2f
    cdef readonly _AdjHelper f2c
    cdef readonly _AdjHelper c2n
    cdef readonly _AdjHelper f2n

    def __init__(self,
                int dim,
                int ncells,
                np.ndarray[NP_FLOAT,ndim=2] x,
                np.ndarray[NP_FLOAT] h,
                _AdjHelper c2f,
                _AdjHelper f2c,
                _AdjHelper c2n,
                _AdjHelper f2n) -> None:
        
        self.dim = dim
        self.ncells = ncells
        # the cython memory view assume that the underlining data is writable but it might happen that the input x is readonly. 
        #By applying np.array(x) it is now writable. But in fact I don't really need to write any thing into x
        self.x = np.array(x) 
        self.h = np.array(h)
        self.c2f = c2f
        self.f2c = f2c
        self.c2n = c2n
        self.f2n = f2n

cdef set[NP_INT] _get_patch_cells(int e, _MeshHelper m):
    """
        Collect the patch of cells surrounding e, this includes e
    """
    cdef set[NP_INT] patch

    facets = m.c2f.links(e)

    for f in facets:
        for ee in m.f2c.links(f):
            patch.insert(ee)
    
    return patch
    
cdef void _compute_virtual_points(int e, _MeshHelper m, 
                            np.ndarray[NP_FLOAT] vh, 
                            np.ndarray[NP_FLOAT] v_e, 
                            np.ndarray[NP_FLOAT,ndim=2] pts_e):
    """
        Compute the virtual points associated to cell e
    """
    cdef _AdjHelper c2f = m.c2f
    cdef _AdjHelper f2c = m.f2c
    cdef _AdjHelper c2n = m.c2n
    cdef _AdjHelper f2n = m.f2n
    cdef int ne = 0

    #get the complete set of nodes
    nodes_e = c2n.links(e)

    for f in c2f.links(e):
        if f2c.links(f).shape[0] == 1:
            #if the current facet is an exterior facet
            
            nodes_f = f2n.links(f) #nodes on f
            node_op = np.setdiff1d(nodes_e,nodes_f) # opposite node
            
            #compute virtual points
            v_e[ne] = vh[nodes_f[0]]+vh[nodes_f[1]]-vh[node_op[0]]
            pts_e[ne,:] = np.array(m.x[nodes_f[0],:]) + np.array(m.x[nodes_f[1],:]) - np.array(m.x[node_op[0],:])

            ne += 1

cdef set[NP_INT] _collect_nodes(set[NP_INT] patch, _MeshHelper m):
    """
        Collect all the node if contained in the patch of cells
    """
    cdef set[NP_INT] nodes
    for e in patch:
        for n in m.c2n.links(e):
            nodes.insert(n)
    return nodes

cdef _interp_on_cells(Py_ssize_t e,
                    np.ndarray[NP_FLOAT,ndim=1] vh,
                    np.ndarray[NP_FLOAT,ndim=2] tab,
                    _MeshHelper m,
                    np.ndarray[NP_FLOAT,ndim=1] dof):
    
    cdef np.ndarray[NP_FLOAT,ndim=1] v_e = np.zeros((6,), dtype=np.float64)
    cdef np.ndarray[NP_FLOAT,ndim=2] pts_e = np.zeros((6,3), dtype=np.float64)
    cdef int ne = 0 #local node index
    
    patch = _get_patch_cells(e, m)
    nodes = _collect_nodes(patch, m)
    
    for ng in nodes: # for known nodes, collect dof vh and interpolation points
        v_e[ne] = vh[ng]
        pts_e[ne,:] = m.x[ng,:]
        ne += 1

    if nodes.size() < 6:
        #compute virtual points and start inserting at ne
        _compute_virtual_points(e, m, vh, v_e[ne:], pts_e[ne:,:]) 

    tab_e = tab[6*e:6*(e+1),:]
    h_e = m.h[e]

    Ve = _Vandermonde((pts_e-tab_e[0,:])/h_e)
    Vtab = _Vandermonde((tab_e-tab_e[0,:])/h_e)

    dof[6*e:6*(e+1)],_,_,_ = Vtab@np.linalg.lstsq(Ve, v_e)

cdef _Vandermonde(np.ndarray[NP_FLOAT,ndim=2] pts):
    return np.array([[p[0]**2, p[1]**2, p[0] * p[1], p[0], p[1], 1] for p in pts])

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _quad_interp(np.ndarray[NP_FLOAT,ndim=1] vh, 
                np.ndarray[NP_FLOAT,ndim=2] tab,
                _MeshHelper m):

    cdef Py_ssize_t e = 0
    cdef int ncells = m.ncells
    cdef np.ndarray[NP_FLOAT] dof = np.zeros((6*m.ncells,), dtype=np.float64)

    for e in range(ncells):
        _interp_on_cells(e, vh, tab, m, dof)

    return dof




