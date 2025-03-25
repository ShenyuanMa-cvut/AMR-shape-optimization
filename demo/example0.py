from AMR_TO import elasticity as elas
from AMR_TO.microstructure import solve_microstructure_batch
from AMR_TO.interpolation import averaging_kernel

import dolfinx,ufl
import pyvista

import pickle
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np

from utils import *

def marking(eta,threshold):
    argsort = np.argsort(eta)[::-1]
    size = argsort.size
    return argsort[:int(size*threshold)]

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
    lc = .05
    new_mesh = generate_mesh(points,load_points_raw,lc=lc) #generate a very coarse mesh

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
    solver.update_mesh(new_mesh)
    Ave = averaging_kernel(new_mesh)

    #initialize the design
    ncells0 = new_mesh.topology.index_map(dim).size_local
    A0h = np.zeros((ncells0,edim,edim))
    A0h[:] = A0[None,:,:]
    solver.Ah.x.array[:] = A0h.reshape(-1)
    solver.thetah.x.array[:] = 1.
    
    compl,vol = solver.solve_all() #initial solve

    l = 1e-1
    delta = 1e-4
    itermax = 100

    history = {'compl':[],'vol':[],'oc':[]}
    i = 0
    p = 1.05
    ncells = ncells0

    while ncells <= 24000:
        solver.update_mesh(new_mesh)
        solver.solve_all()
        Ave = averaging_kernel(solver.mesh)
        print(f"Refinement : new mesh has {solver.ncells} cells")
        
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
            solver.Ah.x.array[:] = np.linalg.inv(A0inv[None,:,:]+(1-te[:,None,None])/te[:,None,None]*FAinv).reshape(-1) 
            
            #update objective
            compl,vol = solver.solve_all()
            oc = solver.compute_oc(FAinv, l)
            print(f"{k} : {sum(compl)} {vol} {oc}")

            history['compl'].append(sum(compl))
            history['vol'].append(vol)
            history['oc'].append(oc)

            if k == 9:
                #fit a power law to oc
                k_ = k + 1
                a,b,c = fit_power_law(range(2,k_+1),history['oc'][-k_+1:])
            elif k > 10:
                if history['oc'][-1] <= a*p:
                    break
            
        new_mesh,_,_ = dolfinx.mesh.refine(solver.mesh)
        ncells = new_mesh.topology.index_map(dim).size_local
        i += 1

    #save the final mesh
    topology_vtk, cell_types, x = dolfinx.plot.vtk_mesh(solver.mesh)
    grid = pyvista.UnstructuredGrid(topology_vtk, cell_types, x)
    grid.cell_data['theta'] = solver.thetah.x.array
    grid.cell_data['h'] = solver.mesh.h(solver.dim, np.arange(solver.ncells))
    grid.save(f"data/final_mesh_uni_{ncells0}_{solver.ncells}.vtu")

    #save the obj history
    with open(f"data/history_uni_{ncells0}_{solver.ncells}.pkl", "wb") as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main()