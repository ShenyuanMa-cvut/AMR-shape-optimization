import dolfinx
import pyvista
import gmsh
import numpy as np
from geometry import is_between
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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

def pretty_plot(domain : dolfinx.mesh.Mesh, mark_label = True) -> pyvista.Plotter:
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
    
    pyvista.set_jupyter_backend("static")
    pyvista.start_xvfb()
    p = pyvista.Plotter()
    
    # Extract topology from mesh and create pyvista mesh
    topology_vtk, cell_types, x = dolfinx.plot.vtk_mesh(domain)
    grid = pyvista.UnstructuredGrid(topology_vtk, cell_types, x)
    
    actor_0 = p.add_mesh(grid,style="wireframe", color="k")

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
    
    return p

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
    
def generate_mesh(geo_points : list[tuple[float]], load_points : list[tuple[float]], name= None, lc = None) -> tuple[dolfinx.mesh.Mesh]:
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
            load_vtx = [gmsh.model.geo.addPoint(*l,0.) for l in load_points]
        else:
            vtx = [gmsh.model.geo.addPoint(*g,0.,lc) for g in geo_points]
            load_vtx = [gmsh.model.geo.addPoint(*l,0.,lc) for l in load_points]
        
        pts_draw = vtx + [vtx[0]]
        pts_coord = geo_points + [geo_points[0]]
        
        for tag,point in zip(load_vtx, load_points):
            #for each point (assuming on the boundary) insert it into pts_draw
            ind = insert_to_loop(point, pts_coord)
            if ind > 0:
                pts_draw.insert(ind, tag)
                pts_coord.insert(ind, point)

        lines = [gmsh.model.geo.addLine(pts_draw[i],pts_draw[i+1]) for i in range(len(pts_draw)-1)]
        loop = gmsh.model.geo.addCurveLoop(lines)
        holdall = gmsh.model.geo.addPlaneSurface([loop])        
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [holdall], name="Hold_all")
            # We can then generate a 2D mesh...
        gmsh.model.mesh.generate(2)

        domain0,_,_ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), gdim=2)   
    return domain0