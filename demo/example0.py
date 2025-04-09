from AMR_TO import elasticity as elas
import numpy as np

from utils import *

def post_action(optimizer : elas.Optimization, opt_param):
    return 

def final_action(optimizer : elas.Optimization, opt_param):
    for (i,oc) in zip(optimizer.history['i'],optimizer.history['oc']):
        print(f'iter {i} : {oc}')

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
    mesh0 = generate_mesh(points,load_points_raw,lc=lc) #generate a very coarse mesh

    #marker of dirichlet bc
    def dirichlet(x : np.ndarray) -> np.ndarray[bool]:
        is_left = np.isclose(x[0],0.)
        #is_right = np.isclose(x[0],length)   
        return is_left

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

    #initialize the abstract solver
    solver = elas.ElasticitySolver(dim, [dirichlet,dirichlet],[vn1,vn2],g)

    #initialize optimizer
    optimizer = elas.Optimization("uniform_cantilever", solver, mesh0, mu, lmbda, False)
    
    #optimization parameters
    opt_param = {'l':0.5,
        'delta':1e-4,
        'inner_itermax':30,
        'ncell_max':10000}
    
    optimizer.main_loop(opt_param, post_action=post_action, final_action=final_action)

if __name__ == '__main__':
    main()