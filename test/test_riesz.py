import time

import dolfinx
import ufl
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from AMR_TO.interpolation import riesz_representation

from utils import generate_mesh

def test_riesz_representation():
    #create a mesh
    mesh0 = generate_mesh([(0.,0.),(1.,0.),(1.,0.5),(0.5,0.5),(0.5,1.),(0.,1.)],[],lc=0.1)

    Vh = dolfinx.fem.functionspace(mesh0, ('CG',1))
    Th = dolfinx.fem.functionspace(mesh0, ('DG',0))
    vh = dolfinx.fem.Function(Vh)
    vh.x.array[:] = np.random.uniform(size=vh.x.array.size)
    
    dx = ufl.Measure('dx',mesh0)
    ds = ufl.Measure('ds',mesh0)
    dS = ufl.Measure('dS',mesh0)

    t = ufl.TestFunction(Th)
    L = t*vh*dx+t*ds+ufl.avg(t)*dS
    th = riesz_representation(L) #find a DG0 function such that <th,t>=L(t) \forall t

    sh = dolfinx.fem.Function(Th)
    for k in range(10):
        vh.x.array[:] = np.random.uniform(size=vh.x.array.size)
        th = riesz_representation(L)
        sh.x.array[:] = np.random.uniform(size=sh.x.array.size)
        
        assert np.isclose(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.replace(L, {t:sh}))),
                        dolfinx.fem.assemble_scalar(dolfinx.fem.form(th*sh*dx)))

if __name__ == '__main__':
    test_riesz_representation()