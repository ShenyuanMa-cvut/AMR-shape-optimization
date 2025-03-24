from AMR_TO import elasticity as elas
from AMR_TO.microstructure import solve_microstructure_batch
from AMR_TO.interpolation import averaging_kernel

import ufl
import dolfinx

from mpi4py import MPI
import numpy as np
from matplotlib import pyplot as plt

from utils import *

def marking(eta,threshold):
    argsort = np.argsort(eta)[::-1]
    eta_sort = eta[argsort]
    eta_max = eta_sort[0]
    return argsort[eta_sort >= eta_max*(1-threshold)]

def all_edges(mesh : dolfinx.mesh.Mesh,marked : np.ndarray):
    dim = mesh.topology.dim
    fdim = dim - 1
    return np.unique(dolfinx.mesh.compute_incident_entities(mesh.topology, marked, dim, fdim))

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

    # Define the hold all domain (let's work with polygonal domain and just give the coordinates of vertices
    # in counter-clockwise order
    length,height = 2.,1/3.
    points = [(0., 0.),(length, 0.),(length, height),(0., height)]

    # Define the part of the boundary where loads are applied
    load_points = [[(length/6,0.),(length/3,0.)],[(2*length/3,0.),(5*length/6,0.)]]
    load_points_raw = sum(load_points,start=[])
    mesh0 = generate_mesh(points,load_points_raw,lc=.05) #generate a very coarse mesh

    #marker of dirichlet bc
    def dirichlet(x : np.ndarray) -> np.ndarray[bool]:
        is_left = np.isclose(x[0],0.)
        is_right = np.isclose(x[0],length)   
        return np.logical_or(is_left,is_right)

    #marker of vn1 bc
    def vn1(x : np.ndarray) -> np.ndarray[bool]:
        is_bottom = np.isclose(x[1],0.)
        start,end = load_points[0][0][0],load_points[0][1][0]
        is_seg = np.logical_and(x[0] >= start, x[0] <= end)
        return is_bottom*is_seg

    #marker of vn2 bc
    def vn2(x : np.ndarray) -> np.ndarray[bool]:
        is_bottom = np.isclose(x[1],0.)
        start,end = load_points[1][0][0],load_points[1][1][0]
        is_seg = np.logical_and(x[0] >= start, x[0] <= end)
        return is_bottom*is_seg

    g = [np.array([0.,-0.1]),np.array([0.,-0.1])]

    solver = elas.ElasticitySolver(dim, [dirichlet,dirichlet],[vn1,vn2],g) #intialize the abstract solver
    #update the first mesh
    solver.update_mesh(mesh0)
    Ave = averaging_kernel(mesh0)

    #initialize the design
    ncells = mesh0.topology.index_map(dim).size_local
    A0h = np.zeros((ncells,edim,edim))
    A0h[:] = A0[None,:,:]
    solver.Ah.x.array[:] = A0h.reshape(-1)
    solver.thetah.x.array[:] = 1.
    
    compl,vol = solver.solve_all() #initial solve

    l = 1e-1
    delta = 1e-4
    itermax = 20
    print(f"Initial mesh has {solver.ncells} cells.")

    compl_history = []
    vol_history = []
    oc_history = []

    while solver.ncells <= 15000:
        print("k : comple vol oc")
        for k in range(itermax):
            
            #compute microstructures
            tauhs = [tauh.x.array for tauh in solver.tauhs]
            gtau, FA, My = solve_microstructure_batch(mu, lmbda, tauhs, dim, solver.ncells)
            FAinv = np.linalg.inv(FA)
            
            #update design theta
            solver.thetah.x.array[:] = Ave.dot(np.maximum(np.minimum(1.,np.sqrt(gtau/l)),delta))
            #update Hooke's law
            te = solver.thetah.x.array
            solver.Ah.x.array[:] = np.linalg.inv(A0inv[None,:,:]+(1-te[:,None,None])/te[:,None,None]*FAinv).reshape(-1) #update objective
            compl,vol = solver.solve_all()
            oc = solver.compute_oc(FAinv, l)
            print(f"{k} : {sum(compl)} {vol} {oc}")
            compl_history.append(sum(compl))
            vol_history.append(vol)
            oc_history.append(oc)

        new_mesh,_,_ = dolfinx.mesh.refine(solver.mesh)
        solver.update_mesh(new_mesh)
        solver.solve_all()
        Ave = averaging_kernel(solver.mesh)
        print(f"Refinement : new mesh has {solver.ncells} cells")

    #Visualize the mesh
    p,grid = pretty_plot(solver.mesh, mark_label=False)

    #Plot forces
    arrow_num = 10
    directions = np.zeros((arrow_num,3))
    directions[:,1] = -0.1

    center1 = np.zeros((arrow_num,3))
    center1[:,0] = np.linspace(length/6,length/3,arrow_num)

    center2 = np.zeros((arrow_num,3))
    center2[:,0] = np.linspace(2*length/3,5*length/6,arrow_num)
    
    p.add_arrows(center1,directions,color='red')
    p.add_arrows(center2,directions,color='blue')

    #visualize data

    grid.cell_data['theta'] = solver.thetah.x.array
    grid.set_active_scalars('theta')
    p.add_mesh(grid, cmap='gray', flip_scalars=True)

    p.view_xy()
    p.show()


    plt.plot(compl_history)
    plt.plot(vol_history)
    plt.plot(oc_history)
    plt.show()
if __name__ == '__main__':
    main()