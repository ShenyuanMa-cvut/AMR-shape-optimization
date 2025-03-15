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

def test_interpolation(shape,symm):
    #create a mesh
    Nx,Ny = 128,128
    mesh0= dolfinx.mesh.create_unit_square(comm, Nx,Ny)
    dim = mesh0.topology.dim
    ncells = mesh0.topology.index_map(dim).size_local

    if len(shape)==2 and shape[0]==shape[1]:
        Vh = dolfinx.fem.functionspace(mesh0, ('CG',1, shape, symm))
    else:
        Vh = dolfinx.fem.functionspace(mesh0, ('CG',1, shape))

    #create a random function 
    vh = dolfinx.fem.Function(Vh)
    vh.x.array[:] = np.random.uniform(size=len(vh.x.array))

    start = time.time()
    th = local_quad_interpolation_cy(vh)
    print(f"Time for a local quad interpolation {time.time()-start}(s) on {Nx}*{Ny} mesh")
    print(f"Codimension = {max(Vh.num_sub_spaces,1)}, Ndof = {len(vh.x.array)}, shape = {shape}")
    print()

    #checking
    c2n = mesh0.topology.connectivity(dim,0)
    for e in range(ncells):
        patch = get_patch_cells(e,mesh0)
        cells = [] #computation cells
        nodes = [] #computation nodes

        for ee in patch:
            cells += [ee]*3
            nodes += c2n.links(ee).tolist()

        vh_val = vh.eval(mesh0.geometry.x[nodes], np.array(cells,dtype=np.int32))
        th_val = th.eval(mesh0.geometry.x[nodes], np.array([e]*len(cells),dtype=np.int32))

        assert np.allclose(vh_val,th_val)
        

if __name__ == '__main__':

    test_interpolation((),False)
    test_interpolation((2,),False)
    test_interpolation((2,2),True)
    test_interpolation((2,2),False)
    test_interpolation((3,3),True)
    test_interpolation((3,3),False)
    test_interpolation((3,4),False)