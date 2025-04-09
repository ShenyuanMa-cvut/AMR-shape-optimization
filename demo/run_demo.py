from AMR_TO import elasticity as elas
import numpy as np

from utils import *

def post_action(optimizer : elas.Optimization, opt_param):
    return 

def final_action(optimizer : elas.Optimization, opt_param):
    for (i,oc) in zip(optimizer.history['i'],optimizer.history['oc']):
        print(f'iter {i} : {oc}')

def main():
    filename = "exp/cantilever_uniform.json"
    solver,mesh,opt_param,material = exp_parser(filename)

    
    print(opt_param)
    print(material)

    p,grid = pretty_plot(mesh,False)
    p.add_mesh(grid, style='wireframe')
    p.view_xy()
    p.show()

    return

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