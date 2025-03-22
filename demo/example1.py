from AMR_TO import elasticity as elas
from AMR_TO.microstructure import solve_microstructure_batch
from AMR_TO.interpolation import averaging_kernel

import ufl
import dolfinx

from mpi4py import MPI
import numpy as np

import gmsh
import pyvista

from .utils import *

def main():
    # Define physical quantities
    dim = 2
    fdim = dim - 1
    edim = 6 if dim == 3 else 3

    kappa = 1 #bulk modulus
    mu = 0.5 #shear modulus
    lmbda = kappa-2*mu/dim # Poisson ration
    A0 = Hooke(mu,lmbda,dim)
    A0inv = np.linalg.inv(A0)

if __name__ == '__main__':
    main()