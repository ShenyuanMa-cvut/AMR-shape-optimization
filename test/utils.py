import dolfinx
import ufl
from mpi4py import MPI
import pyvista
import numpy as np

import gmsh

import concurrent.futures as cf
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class _GmshContext:
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

class _AdjHelper:
    def __init__(self, adj:dolfinx.cpp.graph.AdjacencyList_int32):
        """
            A serializable wrapper of the adjacency list.
        """
        self._array = adj.array
        self._offsets = adj.offsets
        
    def links(self, i):
        return self._array[self._offsets[i]:self._offsets[i+1]]

class _MeshHelper:
    def __init__(self,mesh:dolfinx.mesh.Mesh):
        """
            A serializable wrapper of mesh. Because I want to use ray to parallelize some mesh operation
        """
        self._dispatch(mesh)
        
    def _dispatch(self,mesh:dolfinx.mesh.Mesh):
        #geometry
        self.x = mesh.geometry.x
        #topology
        dim = mesh.topology.dim
        fdim = dim-1
        
        self.dim = dim
        
        self.connectivity = dict()
        self.connectivity[(dim,fdim)] = _AdjHelper(mesh.topology.connectivity(dim,fdim))
        self.connectivity[(fdim,dim)] = _AdjHelper(mesh.topology.connectivity(fdim,dim))
        self.connectivity[(dim,0)] = _AdjHelper(mesh.topology.connectivity(dim,0))
        self.connectivity[(fdim,0)] = _AdjHelper(mesh.topology.connectivity(fdim,0))

class _Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class _Segment:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def is_between(self, c):
        # Check if slope of a to c is the same as a to b ;
        # that is, when moving from a.x to c.x, c.y must be proportionally
        # increased than it takes to get from a.x to b.x .

        # Then, c.x must be between a.x and b.x, and c.y must be between a.y and b.y.
        # => c is after a and before b, or the opposite
        # that is, the absolute value of cmp(a, b) + cmp(b, c) is either 0 ( 1 + -1 )
        #    or 1 ( c == a or c == b)

        a, b = self.a, self.b             

        return ((b.x - a.x) * (c.y - a.y) == (c.x - a.x) * (b.y - a.y) and 
                abs(_cmp(a.x, c.x) + _cmp(b.x, c.x)) <= 1 and
                abs(_cmp(a.y, c.y) + _cmp(b.y, c.y)) <= 1)

def _cmp(x,y):
    return int((x>y)-(x<y))

def is_between(a,b,c):
    pa = _Point(a[0],a[1])
    pb = _Point(b[0],b[1])
    pc = _Point(c[0],c[1])

    return _Segment(pa,pb).is_between(pc)

def pretty_plot(mesh : dolfinx.mesh.Mesh, mark_label = True) -> tuple[pyvista.Plotter,pyvista.UnstructuredGrid]:
    """
    Beautiful plot of a mesh
    """
    topology = mesh.topology
    geometry = mesh.geometry
    dim = topology.dim
    fdim = dim-1
    topology.create_entities(1)
    num_cell = topology.index_map(dim).size_local
    num_facet = topology.index_map(fdim).size_local
    num_node = topology.index_map(0).size_local
    
    p = pyvista.Plotter()
    
    # Extract topology from mesh and create pyvista mesh
    topology_vtk, cell_types, x = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology_vtk, cell_types, x)
    
    #actor_0 = p.add_mesh(grid,style="wireframe", color="k")

    if mark_label:
        # mark nodes
        points = geometry.x
        labels_node = [f"{i}" for i in range(num_node)]
        actor_node = p.add_point_labels(points,labels_node,text_color="red",shape=None)
        
        # mark facets
        topology.create_connectivity(fdim,dim)
        midpoints = dolfinx.mesh.compute_midpoints(mesh,fdim,np.arange(num_facet))
        labels_facet = [f"{i}" for i in range(num_facet)]
        actor_facet = p.add_point_labels(midpoints,labels_facet,text_color="blue",shape=None)
    
        #mark cells
        midpoints = dolfinx.mesh.compute_midpoints(mesh,dim,np.arange(num_cell))
        labels_cell = [f"{i}" for i in range(num_cell)]
        actor_cell = p.add_point_labels(midpoints,labels_cell,text_color="black",shape=None)
    
    return p,grid

def Vandermonde(pts):
    return np.array([[p[0]**2, p[1]**2, p[0] * p[1], p[0], p[1], 1] for p in pts])

def mesh_quality(mesh:dolfinx.mesh.Mesh):
    Th = dolfinx.fem.functionspace(mesh, ('DG',0))
    dim = mesh.topology.dim
    ncells = mesh.topology.index_map(dim).size_local
    h = mesh.h(dim, np.arange(ncells,dtype=np.int32)) #diameters of the triangles
    
    t = ufl.TestFunction(Th)
    dx = ufl.Measure("dx",mesh)
    area = dolfinx.fem.assemble_vector(dolfinx.fem.form(t*dx)).array

    return np.min(area/(h**2/4*np.pi))

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

def collect_nodes(patch:list[int],mesh:dolfinx.mesh.Mesh)->list[int]:
    """
        Find the ids of the nodes that are in the patch of cells
    """
    dim = mesh.topology.dim
    c2n = mesh.topology.connectivity(dim,0)
    nodes = set()
    for e in patch:
        nodes.update(c2n.links(e).tolist())
    return list(nodes)

def compute_virtual_points(e:int,mesh:dolfinx.mesh.Mesh,vh:np.ndarray)->list[np.ndarray]:
    """
        For an exterior cell, find the coordinates of virtual points and the value of the virtual points
    """
    dim = mesh.topology.dim
    fdim = dim - 1
    c2f = mesh.topology.connectivity(dim,fdim)
    f2c = mesh.topology.connectivity(fdim,dim)
    c2n = mesh.topology.connectivity(dim,0)
    f2n = mesh.topology.connectivity(fdim,0)

    facets = c2f.links(e)
    nodes = set(c2n.links(e))
    vpts = []
    vvals = []
    for f in facets:
        if len(f2c.links(f))==1: #if f is an exterior edge
            nodes_f = f2n.links(f) # nodes on the facet f
            node_op = nodes.difference(nodes_f).pop() #opposite node of the facet
            
            x_facets = mesh.geometry.x[nodes_f]
            x_op = mesh.geometry.x[node_op]
            vpts.append(x_facets[0]+x_facets[1]-x_op)

            vh_facets = vh[nodes_f]
            vh_op = vh[node_op]
            vvals.append(vh_facets[0]+vh_facets[1]-vh_op)
            
    return vpts,vvals

def _prepare_data_batch(e0 : int, e1 : int,
                        h:np.ndarray,
                        tab:np.ndarray,
                        mesh:dolfinx.mesh.Mesh,
                        vh:np.ndarray)->list[np.ndarray]:
    """
        Prepare the Vandermond matrix and the right hand side for the mesh cell e
    """
    dof = np.zeros((e1-e0,6))
    for e in range(e0,e1):
        he = h[e]
        tabe = tab[6*e:6*(e+1)]
        patch = get_patch_cells(e, mesh)
        nodes_e = collect_nodes(patch, mesh)
        
        pts = mesh.geometry.x[nodes_e]
        vals = vh[nodes_e] # one dof per node for Vh
    
        if len(nodes_e)<6:
            vpts, vvals = compute_virtual_points(e, mesh, vh)
            pts = np.vstack([pts,vpts])
            vals = np.concatenate([vals,vvals])
    
        #transform to the "reference cell" manually this improves the conditionning if the cell is very small
        pts_T = (pts-tabe[0])/he
        tab_T = (tabe-tabe[0])/he
    
        Vtab = Vandermonde(tab_T)
        Vpts = Vandermonde(pts_T)
        dof[e-e0,:] = Vtab@np.linalg.lstsq(Vpts,vals)[0].reshape(-1)
    
    return dof, e0 #multithreading does not have the guarantee to preserve order

def _prepare_batches(ntot, ntask):
    if ntask == 1:
        return [0,ntot]

    batches_size = ntot//ntask
    batches = list(range(0,ntot,batches_size))
    batches[-1] = ntot
    return batches
    

# def _lstsq_batch(A : list[np.ndarray], b : list[np.ndarray], *args):
#     return [np.linalg.lstsq(A_,b_, *args)[0].reshape(-1) for (A_,b_) in zip(A,b)]
    
def local_quad_interp(vh:dolfinx.fem.Function, ntask = None)->dolfinx.fem.Function:
    if ntask is None:
        ntask = os.cpu_count()

    Vh = vh.function_space
    mesh = Vh.mesh
    Th = dolfinx.fem.functionspace(mesh, ('DG',2))
    th = dolfinx.fem.Function(Th)
    
    dim = mesh.topology.dim
    fdim = dim-1
    ncells = mesh.topology.index_map(dim).size_local
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(dim,fdim) #cell to facet
    mesh.topology.create_connectivity(fdim,dim) #facet to cell
    mesh.topology.create_connectivity(dim,0) #cell to nodes
    mesh.topology.create_connectivity(fdim,0) #facet to nodes
    
    h = mesh.h(dim,np.arange(ncells,dtype=np.int32)) #diameter of triangles
    tab = Th.tabulate_dof_coordinates() #tabulate dof coordinates

    batches = _prepare_batches(ncells, ntask) #prepare batches of triangles
    with cf.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_prepare_data_batch, e0, e1, h, tab, mesh, vh.x.array) for (e0,e1) in zip(batches[:-1],batches[1:])]
        res = [f.result() for f in cf.as_completed(futures)] #obtain results

    res.sort(key=lambda r:r[1]) #sort the result because multithreading does not guarantee order
    dof = np.vstack([r[0] for r in res])
    th.x.array[:] = dof.reshape(-1)
    return th

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
        
    with _GmshContext() as gmsh_ctx:
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