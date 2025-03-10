from ._interpolation_helper import _MeshHelper,_AdjHelper,_quad_interp
import dolfinx
import numpy as np

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