import time

import dolfinx
import dolfinx.fem.function
import ufl
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from AMR_TO.interpolation import averaging_kernel

from utils import generate_mesh

def test_averaging():
    #create a mesh
    mesh0 = generate_mesh([(0.,0.),(1.,0.),(1.,0.5),(0.5,0.5),(0.5,1.),(0.,1.)],[],lc=0.05)
    dim = mesh0.topology.dim
    ncells = mesh0.topology.index_map(dim).size_local

    Th = dolfinx.fem.functionspace(mesh0, ('DG',0))
    th = dolfinx.fem.Function(Th)
    th.x.array[:] = np.random.uniform(-1., 1., ncells)

    dx = ufl.Measure('dx', mesh0)
    t = ufl.TestFunction(Th)

    areas = dolfinx.fem.assemble_vector(dolfinx.fem.form(t*dx)) #cell areas
    cell_integral = dolfinx.fem.assemble_vector(dolfinx.fem.form(t*th*dx)) #cell integral

    integral_th = dolfinx.fem.assemble_scalar(dolfinx.fem.form(th*dx)) #domain integral

    assert np.allclose(areas.array*th.x.array, cell_integral.array) # sanity check
    assert np.isclose(integral_th, np.sum(cell_integral.array)) # sanity

    start = time.time()
    Ave = averaging_kernel(mesh0)
    print(f"Time to construct kernel {time.time()-start}(s) on mesh having {ncells} cells")

    th_tilde = dolfinx.fem.Function(Th)
    th_tilde.x.array[:] = Ave.dot(th.x.array)

    integral_th_tilde = dolfinx.fem.assemble_scalar(dolfinx.fem.form(th_tilde*dx))

    assert np.all(th_tilde.x.array <= 1.) #the averaging must preserve bounds
    assert np.all(th_tilde.x.array >= -1.) #the averaging must preserve bounds
    assert np.isclose(integral_th_tilde, integral_th) #the averaging must preserve integral

if __name__ == '__main__':
    test_averaging()