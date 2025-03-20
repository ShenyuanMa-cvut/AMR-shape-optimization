import basix.ufl
import dolfinx.fem.petsc
import ufl.form
from ._interpolation_helper import _MeshHelper,_AdjHelper,_quad_interp
import dolfinx,ufl,basix
import numpy as np
import scipy.sparse as scs
import cvxpy as cvx

def _get_patch_cells(e:int, mesh:dolfinx.mesh.Mesh)->list[int]:
    """
        Find the patch of cells surrounding cell e and including e
    """
    dim = mesh.topology.dim
    fdim = dim - 1

    mesh.topology.create_connectivity(dim,fdim)
    mesh.topology.create_connectivity(fdim,dim)

    c2f = mesh.topology.connectivity(dim,fdim)
    f2c = mesh.topology.connectivity(fdim,dim)
    
    facets = c2f.links(e) #all the facets connected to cell e
    patch = set()
    for f in facets:
        patch.update(f2c.links(f).tolist())
    return list(patch)


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

def averaging_kernel(mesh:dolfinx.mesh.Mesh)->scs.csc_matrix:
    """
        Construct an averaging kernel over the mesh (applied to DG0 functions). The averaging kernel A satisfies the following properties:
        1. is linear
        2. Let th be a (DG0) function and the function A@th (averaged function):
            preserves the integral : \int th dx = \int A@th dx 
            preserves bounds : if a<=th<=b cellwise then a<= A@th <= b
        3. is local :
            Consider the averaged value [A@th]_i = \sum_{j} A_{ij} th_j and A_{ij} is nonzero 0 iff cell i and cell j are neighbors

        let T_i be the area of the cell i then 
            2.1 is equivalent to \sum_i A_{ij} T_i = T_j \forall j (the vector of cell areas is a left eigenvector of A)
            2.2 is implied by A_ij being non zero and being a partition of unity : \sum_j A_ij = 1 \forall i

        We find such A by solving a convex optimization problem. Indeed, if A is the identity then this is trivially a working averaging operator but this is clearly not satisfactory. We choose to minimize \sum_ij A_ij^2 (This choice is arbitrary).
    """
    dim = mesh.topology.dim
    fdim = dim - 1
    mesh.topology.create_connectivity(fdim,dim)
    f2c = mesh.topology.connectivity(fdim,dim)

    #locate the indices of interior facets
    int_facets = np.nonzero((f2c.offsets[1:]-f2c.offsets[:-1]) == 2)[0]
    Ncells = mesh.topology.index_map(dim).size_local
    Nint_facets = int_facets.shape[0]

    Th = dolfinx.fem.functionspace(mesh, ('DG',0)) #cellwise constant functions
    t = ufl.TestFunction(Th)
    dx = ufl.Measure('dx', mesh)

    #cell areas
    area = dolfinx.fem.assemble_vector(dolfinx.fem.form(t*dx)).array

    omega_e = cvx.Variable(Ncells, name='omega_e') #diagonal weights to be found
    omega_f = cvx.Variable((Nint_facets,2), name='omega_f') #off diagonal weights

    partition_unity = [1-o for o in omega_e]
    preserve_integral = [a-a*o for (o,a) in zip(omega_e,area)]

    for k,f in enumerate(int_facets):
        i,j = f2c.links(f)
        ai,aj = area[[i,j]]
        omega_ij,omega_ji = omega_f[k]
        partition_unity[i] -= omega_ij
        partition_unity[j] -= omega_ji
        preserve_integral[i] -= aj*omega_ji
        preserve_integral[j] -= ai*omega_ij
    cons = [p==0 for p in partition_unity]+[p==0 for p in preserve_integral]+[omega_e>=0.]+[omega_f>=0.]
    obj = cvx.Minimize(cvx.norm2(omega_e)**2+cvx.norm2(omega_f.reshape(-1))**2)
    prob = cvx.Problem(obj, cons)
    _=prob.solve(solver=cvx.SCS)

    Ave = scs.dok_matrix((Ncells,Ncells),dtype=np.float64)
    Ave[range(Ncells),range(Ncells)] = omega_e.value

    for k,f in enumerate(int_facets):
        i,j = f2c.links(f)
        Ave[i,j] = omega_f.value[k,0]
        Ave[j,i] = omega_f.value[k,1]
    
    return Ave.tocsr()

def riesz_representation(L:ufl.form):
    """
        Let L be a linear form defined on a finite element space. Find a function vh in that space such that 
        <vh,v>=L(v) forall v
    """
    v = L.arguments()[0] #get the arguments
    Vh = v.ufl_function_space()
    u = ufl.TrialFunction(Vh)
    dx = ufl.Measure('dx',Vh.mesh)
    p = dolfinx.fem.petsc.LinearProblem(u*v*dx, L)
    return p.solve()
