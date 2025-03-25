import basix.ufl
import dolfinx.fem.petsc
import ufl.form
from ._interpolation_helper import _MeshHelper,_AdjHelper,_quad_interp
import dolfinx,ufl,basix
import numpy as np

import scipy.sparse as scs
import cvxopt as co
import mosek

def local_quad_interpolation_cy(vh:dolfinx.fem.Function):
    """
        Assuming vh to be a ('CG',1) function. Find the th discontinuous ('DG',2) function that locally interpolates vh : 
        1. For each interior triangle T, th interpolates vh at the three nodes and the other three nodes of the 3 neighbor triangles
        2. For each triangle having at least one exterior edge, we extend virtually the triangle to the exterior of the mesh by applying central symmetry to the opposite node of each exterior edge with respect to the mid point of the exterior edge and we extend linearly vh outside of the mesh
    """
    Vh = vh.function_space
    mesh = Vh.mesh
    dim = mesh.topology.dim
    fdim = dim-1

    shape = vh.ufl_shape
    symm = Vh.ufl_element().is_symmetric
    num_sub_spaces = Vh.num_sub_spaces
    

    # initialize mesh helper
    mesh.topology.create_connectivity(dim,fdim)
    mesh.topology.create_connectivity(fdim,dim)
    mesh.topology.create_connectivity(dim,0)
    mesh.topology.create_connectivity(fdim,0)

    ncells = mesh.topology.index_map(dim).size_local

    c2f = mesh.topology.connectivity(dim,fdim)
    f2c = mesh.topology.connectivity(fdim,dim)
    c2n = mesh.topology.connectivity(dim,0)
    f2n = mesh.topology.connectivity(fdim,0)

    c2f_helper = _AdjHelper(c2f.array, c2f.offsets)
    f2c_helper = _AdjHelper(f2c.array, f2c.offsets)
    c2n_helper = _AdjHelper(c2n.array, c2n.offsets)
    f2n_helper = _AdjHelper(f2n.array, f2n.offsets)

    m = m = _MeshHelper(mesh.topology.dim,
                ncells,
                mesh.geometry.x,
                mesh.h(dim,np.arange(ncells)),
                c2f_helper,
                f2c_helper,
                c2n_helper,
                f2n_helper)
    
    if len(shape)==2 and shape[0]==shape[1]:
        DG2 = basix.ufl.element('DG',mesh.basix_cell(),2,discontinuous=True,shape=shape,symmetry=symm)
    else:
        DG2 = basix.ufl.element('DG',mesh.basix_cell(),2,discontinuous=True,shape=shape)

    Th = dolfinx.fem.functionspace(mesh, DG2)
    tab = Th.tabulate_dof_coordinates()
    th = dolfinx.fem.Function(Th)

    if num_sub_spaces > 0: # if Vh is a vector valued space
        for i in range(num_sub_spaces): #for each subspace
            _,dof_u = Vh.sub(i).collapse()
            _,dof_t = Th.sub(i).collapse()
            th.x.array[dof_t] = _quad_interp(vh.x.array[dof_u], tab, m)
    else:
        th.x.array[:] = _quad_interp(vh.x.array, tab, m)
    
    return th

def _cell_areas(mesh:dolfinx.mesh.Mesh)->np.ndarray:
    Th = dolfinx.fem.functionspace(mesh, ('DG',0))
    t = ufl.TestFunction(Th)
    dx = ufl.Measure('dx', mesh)
    areas = dolfinx.fem.assemble_vector(dolfinx.fem.form(t*dx))
    return areas.array

def _compute_int_facets_idx(mesh:dolfinx.mesh.Mesh):
    dim=mesh.topology.dim
    fdim = dim-1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, dim)
    f2n = mesh.topology.connectivity(fdim,dim)
    offsets = f2n.offsets
    return np.nonzero((offsets[1:]-offsets[:-1])==2)[0]

def _compute_cons_A(mesh:dolfinx.mesh.Mesh, areas : np.ndarray)->co.spmatrix:
    dim = mesh.topology.dim
    fdim = dim - 1
    ncells = mesh.topology.index_map(dim).size_local
    int_facets = _compute_int_facets_idx(mesh) #index of interior facets
    f2c = mesh.topology.connectivity(fdim,dim)
    
    row = list(range(2*ncells))
    col = list(range(ncells))+list(range(ncells))
    data = [1. for i in range(2*ncells)]

    for k,f in enumerate(int_facets):
        i,j = f2c.links(f)
        i = int(i)
        j = int(j)
        Ai,Aj = areas[[i,j]]
        row += [i,j,ncells+i,ncells+j]
        col += [ncells+2*k, ncells+2*k+1, ncells+2*k+1, ncells+2*k]
        data += [1., 1., float(Aj/Ai), float(Ai/Aj)]
    return co.spmatrix(data, row, col, size=(2*ncells, ncells+2*int_facets.size))
    
def _data_to_kernel(x:co.matrix,mesh:dolfinx.mesh.Mesh):
    dim = mesh.topology.dim
    fdim = dim - 1
    ncells = mesh.topology.index_map(dim).size_local
    int_facets = _compute_int_facets_idx(mesh) #index of interior facets
    f2c = mesh.topology.connectivity(fdim,dim)

    A = scs.dok_matrix((ncells,ncells),dtype=np.float64)
    A[range(ncells),range(ncells)] =  list(x[:ncells])

    idx = f2c.offsets[int_facets][:,None]
    idxx = (f2c.offsets[int_facets+1]-1)[:,None]
    ij = f2c.array[np.hstack((idx,idxx))]
    A[ij[:,0],ij[:,1]] = x[ncells::2]
    A[ij[:,1],ij[:,0]] = x[ncells+1::2]

    # for k,f in enumerate(int_facets):
    #     i,j = f2c.links(f)
    #     A[i,j] = x[ncells+2*k]
    #     A[j,i] = x[ncells+2*k+1]

    return A.tocsr()

def averaging_kernel(mesh:dolfinx.mesh.Mesh)->scs.csr_matrix:
    """
        Compute an averaging kernel that preserves integral and bounds, using convex optimization
    """
    areas = _cell_areas(mesh)
    A = _compute_cons_A(mesh, areas)
    b = co.matrix([1. for k in range(A.size[0])])
    G = co.spmatrix(-1., range(A.size[1]),range(A.size[1]))
    P = -G
    q = co.matrix([0. for k in range(A.size[1])])
    h = co.matrix([0. for k in range(A.size[1])])

    mosek_params = {mosek.iparam.log: 0}  # Suppress logging
    co.solvers.options['mosek']=mosek_params

    sol = co.solvers.qp(P, q, G, h, A, b,solver='mosek')

    return _data_to_kernel(sol['x'],mesh)

def riesz_representation(L:ufl.form):
    """
        Let L be a linear form defined on a finite element space. Find a function vh in that space such that 
        <vh,v>=L(v) forall v
    """
    v = L.arguments()[0] #get the arguments
    Vh = v.ufl_function_space()
    u = ufl.TrialFunction(Vh)
    dx = ufl.Measure('dx',Vh.mesh)
    p = dolfinx.fem.petsc.LinearProblem(ufl.inner(u,v)*dx, L)
    return p.solve()
