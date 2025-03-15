# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False,wraparound=False, nonecheck=False

from libcpp.set cimport set
from libcpp.iterator cimport insert_iterator

cdef extern from "<iterator>" namespace "std" nogil:
    insert_iterator[set[int]] inserter(set[int]& container, set[int].iterator pos)

cdef extern from "<algorithm>" namespace "std" nogil:
    OutputIt set_difference[InputIt1, InputIt2, OutputIt](
        InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt out) except +

from cython.operator cimport dereference as deref, preincrement as inc
cimport cython

import numpy as np
cimport numpy as np

from scipy.linalg.cython_lapack cimport dgels

ctypedef np.int32_t NP_INT
ctypedef np.float64_t NP_FLOAT

@cython.final
cdef class _AdjHelper():
    cdef int[:] _array
    cdef int[:] _offsets

    def __init__(self, np.ndarray[NP_INT,ndim=1] array, np.ndarray[NP_INT,ndim=1] offsets):
        self._array = np.array(array,dtype=np.int32)
        self._offsets = np.array(offsets,dtype=np.int32)

    cdef set[int] links(self, int i) nogil:
        cdef set[int] res = set[int]()
        cdef Py_ssize_t k_

        for k_ in range(self._offsets[i],self._offsets[i+1]):
            res.insert(self._array[k_])

        return res

# NP_FLOAT[:] is a memoryview and is a c/c++ thing
# It is understood as NP_FLOAT compactly stored in the memory that does not have "vector space" structure
# I add cython.final decorator to say that the class _meshHelper cannot be inherited (in Python), so it can be used in nogil safely
@cython.final
cdef class _MeshHelper():
    cdef int dim
    cdef int ncells
    cdef NP_FLOAT[:,:] x
    cdef NP_FLOAT[:] h
    cdef _AdjHelper c2f
    cdef _AdjHelper f2c
    cdef _AdjHelper c2n
    cdef _AdjHelper f2n

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
        # the cython memory view assume that the underlining data is writable but it might happen that the input x is. 
        #By applying np.array(x) it is now writable. But in fact I don't really need to write any thing into x
        self.x = np.array(x) 
        self.h = np.array(h)
        self.c2f = c2f
        self.f2c = f2c
        self.c2n = c2n
        self.f2n = f2n

cdef set[int] _libcpp_set_difference(set[int] A, set[int] B) nogil:
    """
    Return a new set[int] that is A \ B using std::set_difference.
    A and B must be sorted containers (true for std::set).
    """
    cdef set[int] result = set[int]()
    set_difference(A.begin(), A.end(), B.begin(), B.end(), inserter(result, result.begin()))
    return result

cdef void _set_2_list(set[int] A, int * L) nogil:
    cdef set[int].iterator it = A.begin()
    cdef Py_ssize_t _i = 0

    while it != A.end():
        L[_i] = deref(it)
        _i += 1
        inc(it)

cdef set[NP_INT] _get_patch_cells(Py_ssize_t e, _MeshHelper m) nogil:
    #Collect the patch of cells surrounding e, this includes e
    cdef set[NP_INT] patch
    cdef Py_ssize_t ee
    cdef Py_ssize_t f

    for f in m.c2f.links(e):
        for ee in m.f2c.links(f):
            patch.insert(ee)
    return patch
    
cdef void _compute_virtual_points(int e, _MeshHelper m, 
                            NP_FLOAT[:] vh, 
                            NP_FLOAT* v_e, 
                            NP_FLOAT* pts_e) nogil:
    #Compute the virtual points associated to cell e
    
    cdef set[int] nodes_e,nodes_f,node_op
    cdef int[2] nodes_f_l
    cdef int[1] node_op_l
    cdef NP_FLOAT[:,:] x = m.x
    cdef Py_ssize_t ne = 0
    cdef Py_ssize_t _f,_n,_k
    cdef NP_FLOAT _v,_p = 0.

    #get the complete set of nodes
    for _n in m.c2n.links(e):
        nodes_e.insert(_n)

    for _f in m.c2f.links(e):
        if m.f2c.links(_f).size() == 1:#if the current facet is an exterior facet
            
            #collect nodes on the facet
            nodes_f = m.f2n.links(_f)    

            #find the opposite node to the current facet
            node_op = _libcpp_set_difference(nodes_e,nodes_f)

            #convert to list to allow referencing
            _set_2_list(nodes_f,nodes_f_l)
            _set_2_list(node_op,node_op_l)

            #compute virtual points
            v_e[ne] = vh[nodes_f_l[0]]+vh[nodes_f_l[1]]-vh[node_op_l[0]]
            
            for _k in range(3):
                pts_e[3*ne+_k] = x[nodes_f_l[0],_k] + x[nodes_f_l[1],_k] - x[node_op_l[0],_k]

            ne += 1
            nodes_f.clear()
            node_op.clear()

cdef set[NP_INT] _collect_nodes(set[NP_INT] patch, _MeshHelper m) nogil:
    #Collect all the node if contained in the patch of cells
    cdef set[NP_INT] nodes
    cdef Py_ssize_t e
    cdef Py_ssize_t n

    for e in patch:
        for n in m.c2n.links(e):
            nodes.insert(n)
    return nodes

cdef void _Vandermonde(NP_FLOAT* pts, NP_FLOAT* V, Py_ssize_t N, Py_ssize_t stride_row, Py_ssize_t stride_col) nogil: #create the Vandermonde matrix
    cdef Py_ssize_t _i,idx
    cdef NP_FLOAT x,y

    for _i in range(N):
        x = pts[3*_i+0]
        y = pts[3*_i+1]

        idx = _i*stride_row
        V[idx+0*stride_col] = x*x
        V[idx+1*stride_col] = y*y
        V[idx+2*stride_col] = x*y
        V[idx+3*stride_col] = x
        V[idx+4*stride_col] = y
        V[idx+5*stride_col] = 1.0

cdef void _least_square(NP_FLOAT* A, NP_FLOAT* b, NP_FLOAT* res, int nrow, int ncol, int stride_row, int stride_col) nogil: #here I always solve 6 by 6 system
    cdef NP_FLOAT[6*6] a_flat
    cdef NP_FLOAT[6] b_flat
    cdef int _i,_j

    cdef char trans = b'N'
    cdef int m = nrow
    cdef int n = ncol
    cdef int nrhs = 1
    cdef int lda = max(1, m)
    cdef int ldb = max(m, n)
    cdef int info
    
    # work space
    cdef int lwork = 64
    cdef NP_FLOAT[64] work


    #get A,b into Fortran format
    for _j in range(6):
        for _i in range(6):
            a_flat[6*_j+_i] = A[_i*stride_row+_j*stride_col]
        b_flat[_j] = b[_j]

    dgels(&trans, &m, &n, &nrhs, &a_flat[0], &lda, &b_flat[0], &ldb, &work[0], &lwork, &info)

    for _i in range(6):
        res[_i] = b_flat[_i]

cdef void _compute_coefficient(NP_FLOAT* pts, NP_FLOAT* val, NP_FLOAT* coeff) nogil:
    cdef NP_FLOAT[6][6] V_array

    _Vandermonde(pts, &V_array[0][0], 6, 6, 1)
    _least_square(&V_array[0][0], val, coeff, 6, 6, 6, 1)

cdef void _find_dof(NP_FLOAT* tab, NP_FLOAT* coeff, NP_FLOAT[:] dof) nogil:
    cdef NP_FLOAT[6][6] V_array

    cdef NP_FLOAT s = 0.
    cdef Py_ssize_t _i,_j
    
    _Vandermonde(tab, &V_array[0][0], 6, 6, 1)
    for _i in range(6):
        for _j in range(6):
            s += V_array[_i][_j]*coeff[_j]
        dof[_i] = s
        s = 0.

cdef void _recenter(NP_FLOAT* pts, NP_FLOAT* p0, NP_FLOAT h, NP_FLOAT* res) nogil:
    cdef Py_ssize_t _i,_j
    for _i in range(6):
        for _j in range(3):
            res[3*_i+_j] = (pts[3*_i+_j]-p0[_j])/h

cdef void _interp_on_cells(Py_ssize_t e,
                    NP_FLOAT[:] vh,
                    NP_FLOAT* tab,
                    _MeshHelper m,
                    NP_FLOAT[:] dof) noexcept nogil:
    
    cdef NP_FLOAT[6] v_e_array, coeff_e_array
    cdef NP_FLOAT[6][3] pts_e_array,pts_T_array,tab_T_arrar #recentered array
    
    cdef NP_FLOAT* v_e = &v_e_array[0]
    cdef NP_FLOAT* coeff_e = &coeff_e_array[0]
    cdef NP_FLOAT* tab_e = &tab[6*3*e]

    cdef NP_FLOAT* pts_e = &pts_e_array[0][0]
    cdef NP_FLOAT* pts_T = &pts_T_array[0][0]
    cdef NP_FLOAT* tab_T = &tab_T_arrar[0][0]

    cdef NP_FLOAT h_e = m.h[e]
    cdef Py_ssize_t ne = 0 #local node index
    cdef Py_ssize_t ng,_k,_i,_j
    
    patch = _get_patch_cells(e, m)
    nodes = _collect_nodes(patch, m)
    
    for ng in nodes: # for physical nodes, collect dof vh and interpolation points
        v_e[ne] = vh[ng]
        for _k in range(3):
            pts_e[3*ne+_k] = m.x[ng,_k]
        ne += 1

    if nodes.size() < 6:#compute virtual points and start inserting at ne
        _compute_virtual_points(e, m, vh, &v_e[ne], &pts_e[3*ne])
    
    _recenter(pts_e,tab_e,h_e,pts_T) #recenter the interp point to the reference cell
    _recenter(tab_e,tab_e,h_e,tab_T) #recenter the tabulation point to the reference cell

    _compute_coefficient(pts_T, v_e, coeff_e) #compute the interpolating coefficients 
    _find_dof(tab_T, coeff_e, dof[6*e:6*e+6]) #compute the dof's

def _quad_interp(np.ndarray[NP_FLOAT,ndim=1] vh_array, 
                np.ndarray[NP_FLOAT,ndim=2] tab_array,
                _MeshHelper m):

    cdef Py_ssize_t e = 0
    cdef Py_ssize_t ncells = m.ncells
    cdef np.ndarray[NP_FLOAT] dof_array = np.zeros((6*m.ncells,), dtype=np.float64)
    cdef NP_FLOAT[:] dof = dof_array
    cdef NP_FLOAT[:] vh = vh_array
    cdef NP_FLOAT* tab = &tab_array[0,0]

    for e in range(ncells):
        _interp_on_cells(e, vh, tab, m, dof)

    return dof_array




