import dolfinx
import pyvista
import gmsh
import numpy as np
import scipy.optimize as so
from geometry import is_between
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import json

import AMR_TO.elasticity as elas

class GmshContext:
    """Implement a gmsh context so that I don't forget to finalize the gmsh each time"""
    def __init__(self, args=None):
        """Initialize GmshContext with optional arguments."""
        self.args = args if args is not None else []

    def __enter__(self):
        """Initialize gmsh when entering the context."""
        gmsh.initialize(self.args)
        return self  # Allow using "with GmshContext() as gmsh_ctx:"

    def __exit__(self, exc_type, exc_value, traceback):
        """Finalize gmsh when exiting the context."""
        gmsh.finalize()

def pretty_plot(domain : dolfinx.mesh.Mesh, mark_label = True) -> tuple[pyvista.Plotter,pyvista.UnstructuredGrid]:
    """
    Beautiful plot of a mesh
    """
    topology = domain.topology
    geometry = domain.geometry
    dim = topology.dim
    fdim = dim - 1
    
    topology.create_entities(1)
    num_cell = topology.index_map(dim).size_local
    num_facet = topology.index_map(fdim).size_local
    num_node = topology.index_map(0).size_local
    
    p = pyvista.Plotter()
    
    # Extract topology from mesh and create pyvista mesh
    topology_vtk, cell_types, x = dolfinx.plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology_vtk, cell_types, x)
    
    #actor_0 = p.add_mesh(grid,style="wireframe", color="k")

    if mark_label:
        # mark nodes
        points = geometry.x
        labels_node = [f"{i}" for i in range(num_node)]
        actor_node = p.add_point_labels(points,labels_node,text_color="red",shape=None)
        
        # mark facets
        topology.create_connectivity(fdim,dim)
        midpoints = dolfinx.mesh.compute_midpoints(domain,fdim,np.arange(num_facet))
        labels_facet = [f"{i}" for i in range(num_facet)]
        actor_facet = p.add_point_labels(midpoints,labels_facet,text_color="blue",shape=None)
    
        #mark cells
        midpoints = dolfinx.mesh.compute_midpoints(domain,dim,np.arange(num_cell))
        labels_cell = [f"{i}" for i in range(num_cell)]
        actor_cell = p.add_point_labels(midpoints,labels_cell,text_color="black",shape=None)
    
    return p,grid

# Hooke's law
def Hooke(mu,lmbda,d):
    """
    Return the Hooke's law tensor in engineering notation for dimension d=2 or 3
    """
    if d==2:
        return np.array([[2*mu+lmbda,lmbda,0],
                         [lmbda,2*mu+lmbda,0],
                         [0,0,2*mu]])
    elif d==3:
        return np.array([[2*mu+lmbda,lmbda,lmbda,0,0,0],
                         [lmbda,2*mu+lmbda,lmbda,0,0,0],
                         [lmbda,lmbda,2*mu+lmbda,0,0,0],
                         [0,0,0,2*mu,0,0,],
                         [0,0,0,0,2*mu,0],
                         [0,0,0,0,0,2*mu]])

def insert_to_loop(pts : tuple[float], loop_pts : list[tuple[float]]) -> int:
    """
        find the index where pts should be inserted into.
        pts : the coordinates of the points
        loop_pts : a loop of points in counterclockwise direction. There is : loop_pts[-1]=loop_pts[0]
    """
    for i in range(len(loop_pts)-1):
        p1 = loop_pts[i]
        p2 = loop_pts[i+1]
        if is_between(p1,p2,pts):
            if np.all(np.isclose(p1,pts)) or np.all(np.isclose(p2,pts)):
                return -1
            return i+1
    
def generate_mesh(geo_points : list[tuple[float]], name= None, lc = None) -> tuple[dolfinx.mesh.Mesh]:
    """
        Generate the initial mesh.
        geo_points : a list of coordinates of vertices of a polygon without self loop. Ordered in counterclockwise direction.
        load_points : points defining the loaded part of the boundaries.
    """
    if name is None:
        name = "domain0"
        
    with GmshContext() as gmsh_ctx:
        gmsh.model.add(name)
        # add points of domain definition
        if lc is None: 
            vtx = [gmsh.model.geo.addPoint(*g,0.) for g in geo_points]
        else:
            vtx = [gmsh.model.geo.addPoint(*g,0.,lc) for g in geo_points]
        
        pts_draw = vtx + [vtx[0]]

        lines = [gmsh.model.geo.addLine(pts_draw[i],pts_draw[i+1]) for i in range(len(pts_draw)-1)]
        loop = gmsh.model.geo.addCurveLoop(lines)
        holdall = gmsh.model.geo.addPlaneSurface([loop])        
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [holdall], name="Hold_all")
            # We can then generate a 2D mesh...
        gmsh.model.mesh.generate(2)

        domain0,_,_ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), gdim=2)   
    return domain0

def fit_power_law(x,y):
    def func(x,a,b,c):
        return a+b*np.power(x, c)
    
    def jac(x,a,b,c):
        da = np.ones_like(x)
        db = np.power(x,c)
        dc = np.power(x,c)*np.log(x)
        return np.array([da,db,dc]).T

    popt = so.curve_fit(func, np.array(x), np.array(y), [1.,1.,-1.], jac=jac)[0]
    return popt

def exp_parser(filename)->tuple[elas.ElasticitySolver,dolfinx.mesh.Mesh,dict,dict]:
    """
        Parse the file provided in filename and return:
        Elasticity solver
        initial mesh
        opt_param
        material
    """
    
    #load config file
    with open(filename) as f:
        exp = json.load(f)
    
    #exp name
    name = exp['name']

    #build mesh
    points = exp['geometry']['points']
    segs = exp['geometry']['segments']
    lc = exp['geometry'].get('lc',None)
    mesh = generate_mesh(points, name=name,lc=lc)

    #opt_param
    opt_param = exp["opt_param"]

    #material
    material = exp["material"]

    return None,mesh,opt_param,material


# # Define physical quantities
# dim = 2
# fdim = dim - 1
# edim = 6 if dim == 3 else 3

# kappa = 1 #bulk modulus
# mu = 0.5 #shear modulus
# lmbda = kappa-2*mu/dim # Poisson ration

# # Define the hold all domain (let's work with polygonal domain and just give the coordinates of vertices
# # in counter-clockwise order
# length,height = 2.,1/3.
# points = [(0., 0.),(length, 0.),(length, height),(0., height)]

# # Define the part of the boundary where loads are applied
# load_points = [[(length/6,0.),(length/3,0.)],[(2*length/3,0.),(5*length/6,0.)]]
# load_points_raw = sum(load_points,start=[])
# lc = .05
# mesh0 = generate_mesh(points,load_points_raw,lc=lc) #generate a very coarse mesh

# #marker of dirichlet bc
# def dirichlet(x : np.ndarray) -> np.ndarray[bool]:
#     is_left = np.isclose(x[0],0.)
#     #is_right = np.isclose(x[0],length)   
#     return is_left

# #marker of vn1 bc
# def vn1(x : np.ndarray) -> np.ndarray[bool]:
#     is_bottom = np.isclose(x[1],0.)
#     start,end = load_points[0][0][0],load_points[0][1][0]
#     is_seg = np.logical_and(x[0] >= start, x[0] <= end)
#     return is_bottom*is_seg

# #marker of vn2 bc
# def vn2(x : np.ndarray) -> np.ndarray[bool]:
#     is_bottom = np.isclose(x[1],0.)
#     start,end = load_points[1][0][0],load_points[1][1][0]
#     is_seg = np.logical_and(x[0] >= start, x[0] <= end)
#     return is_bottom*is_seg

# g = [np.array([0.,-0.1]),np.array([0.,-0.1])]

# #initialize the abstract solver
# solver = elas.ElasticitySolver(dim, [dirichlet,dirichlet],[vn1,vn2],g)