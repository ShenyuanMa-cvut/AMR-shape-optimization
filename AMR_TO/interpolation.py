from ._interpolation_helper import _MeshHelper,_AdjHelper,_quad_interp
import dolfinx,ufl
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
    Vh = vh.function_space
    mesh = Vh.mesh
    dim = mesh.topology.dim
    fdim = dim-1

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

    Th = dolfinx.fem.functionspace(mesh, ('DG',2))
    tab = Th.tabulate_dof_coordinates()

    th = dolfinx.fem.Function(Th)
    th.x.array[:] = _quad_interp(vh.x.array, tab, m)
    return th

def averaging_kernel(mesh:dolfinx.mesh.Mesh)->scs.csc_matrix:
    """
        Construct an averaging kernel over the mesh (applied to DG0 functions). The averaging kernel A satisfies the following properties:
        1. is linear
        2. Let th be a (DG0) function and the function A@th (averaged function):
            preserves the integral
            preserves bounds (if a<=th<=b cellwise then a<= A@th <= b)

        We find such A by solving a convex optimization problem.
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
    _=prob.solve()

    Ave = scs.dok_matrix((Ncells,Ncells),dtype=np.float64)
    Ave[range(Ncells),range(Ncells)] = omega_e.value

    for k,f in enumerate(int_facets):
        i,j = f2c.links(f)
        Ave[i,j] = omega_f.value[k,0]
        Ave[j,i] = omega_f.value[k,1]
    
    return Ave.tocsr()
