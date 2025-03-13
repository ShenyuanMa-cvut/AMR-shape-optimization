import dolfinx
import ufl
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from AMR_TO.interpolation import local_quad_interpolation_cy

import time

def get_patch_cells(e:int, mesh:dolfinx.mesh.Mesh)->list[int]:
    """
        Find the patch of cells surrounding cell e and including e
    """
    dim = mesh.topology.dim
    fdim = dim - 1
    c2f = mesh.topology.connectivity(dim,fdim)
    f2c = mesh.topology.connectivity(fdim,dim)
    
    facets = c2f.links(e) #all the facets connected to cell e
    patch = set()
    for f in facets:
        patch.update(f2c.links(f).tolist())
    return list(patch)

def test_interpolation():
    #create a mesh
    Nx,Ny = 128,128
    mesh0= dolfinx.mesh.create_unit_square(comm, Nx,Ny)
    dim = mesh0.topology.dim
    ncells = mesh0.topology.index_map(dim).size_local

    #create a function
    Vh = dolfinx.fem.functionspace(mesh0, ('CG',1))
    vh = dolfinx.fem.Function(Vh)
    xu = ufl.SpatialCoordinate(mesh0)
    sin_exp = dolfinx.fem.Expression(ufl.cos(2*np.pi*(xu[0]+xu[1]))*2, Vh.element.interpolation_points())
    vh.interpolate(sin_exp)

    start = time.time()
    th = local_quad_interpolation_cy(vh)
    print(f"Time for a local quad interpolation {time.time()-start}(s) on {Nx}*{Ny} mesh")

    #checking
    c2n = mesh0.topology.connectivity(dim,0)
    for e in range(ncells):
        patch = get_patch_cells(e,mesh0)
        nodes_e = np.unique(sum((c2n.links(ee).tolist() for ee in patch),[]))
        vals = vh.x.array[nodes_e]
        xe = mesh0.geometry.x[nodes_e]
        vals_interp = th.eval(xe,np.ones(xe.shape[0],dtype=np.int32)*e).reshape(-1)
        assert np.allclose(vals,vals_interp)

if __name__ == '__main__':
    test_interpolation()