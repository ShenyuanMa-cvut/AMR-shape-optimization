import ufl
from basix.ufl import element
import dolfinx
from dolfinx.fem import petsc

from mpi4py import MPI
import numpy as np
import ufl.constant

from .interpolation import local_quad_interpolation_cy,riesz_representation,_cell_areas

def _epsilon(v : any ,dim : int):
    """
    Strain in engineering notation
    """
    sqrt2 = ufl.sqrt(2)
    e = ufl.sym(ufl.grad(v))
    if dim == 2:
        return ufl.as_vector([e[0,0],e[1,1],sqrt2*e[0,1]])
    elif dim == 3:
        return ufl.as_vector([e[0,0],e[1,1],e[2,2],sqrt2*e[0,1],sqrt2*e[0,2],sqrt2*e[1,2]])
    
def _svecT(tau, dim):
    """
        Let tau be the stress in engineering notation, return tau in matrix representation
    """
    sqrt2 = ufl.sqrt(2)
    if dim == 2:
        tau11 = tau.sub(0)
        tau22 = tau.sub(1)
        tau12 = tau.sub(2)/sqrt2
        return ufl.as_tensor([[tau11,tau12],[tau12,tau22]])
    elif dim == 3:
        tau11,tau22,tau33 = tau.sub(0),tau.sub(1),tau.sub(2)
        tau12,tau13,tau23 = tau.sub(3)/sqrt2,tau.sub(4)/sqrt2,tau.sub(5)/sqrt2

        return ufl.as_tensor([[tau11,tau12,tau13],[tau12,tau22,tau23],[tau13,tau23,tau33]])

def _interpolate(Vh_to : dolfinx.fem.FunctionSpace, vh_from : dolfinx.fem.Function, match=True) -> dolfinx.fem.Function:
    """
    Interpolate the function vh (in Vh_from) to the function space Vh_to. match is True if Vh_to and Vh_from are defined over the same mesh
    """
    vh_to = dolfinx.fem.Function(Vh_to)
    if match:
        vh_to.interpolate(vh_from)
    else:
        mesh_to = Vh_to.mesh
        dim = mesh_to.topology.dim
        ncell_to = mesh_to.topology.index_map(dim).size_local
        cells = np.arange(ncell_to, dtype=np.int32)
        interp_data = dolfinx.fem.create_interpolation_data(Vh_to,vh_from.function_space,cells)
        vh_to.interpolate_nonmatching(vh_from,cells,interp_data)

    return vh_to

class _AbstractElasticityForm():
    """
    This class implements the abstract bilinear form int_Omega A e(u):e(v) d x used in the linear elasticity    
    """
    def __init__(self, dim : int):
        self.dim = dim # space dimension
        self.edim = 6 if dim == 3 else 3 # dimension of engineering vector
        
        # finite elements
        self.u_el = element('Lagrange','triangle',1,shape=(self.dim,)) # FE displacement
        self.x_el = element('Lagrange','triangle',1,shape=(self.dim,)) # FE space coordinates
        self.A_el = element('Lagrange','triangle',0,shape=(self.edim,self.edim),discontinuous=True) #FE Hooke's law
        self.tau_el = element('Lagrange','triangle',0,shape=(self.edim,),discontinuous=True) #FE stress
        self.theta_el = element('Lagrange','triangle',0,discontinuous=True) #FE density

        #abstract mesh representation
        self.mesh_abs = ufl.Mesh(self.x_el)
        dx = ufl.Measure('dx', self.mesh_abs)

        #abstract function spaces
        self.V = ufl.FunctionSpace(self.mesh_abs,self.u_el)
        self.Q = ufl.FunctionSpace(self.mesh_abs,self.A_el)

        #abstract form
        self.u,self.v = ufl.TrialFunction(self.V),ufl.TestFunction(self.V)
        self.A = ufl.Coefficient(self.Q)
        a = ufl.inner(self.A*_epsilon(self.u,self.dim), _epsilon(self.v,self.dim))*dx
        self.a = dolfinx.fem.compile_form(MPI.COMM_WORLD, a)

class ElasticitySolver(_AbstractElasticityForm):
    def __init__(self, dim, dlocators, vnlocators, loads):
        """
        dim : the dimension of the physical domain
        dlocators : a list of functions that locates the coordinates of dof subject to dirichlet bc
        vnlocators : a list of functions that locates the coordinates where vn bc is applied
        loads : the constant values of vn bc       
        """

        super().__init__(dim)
        self.dlocators = dlocators
        self.vnlocators = vnlocators
        self.loads = loads

    def update_mesh(self, mesh : dolfinx.mesh.Mesh):
        """
        Update or assign for the first time the computation mesh        
        """
        self.mesh = mesh
        self.cell_areas = _cell_areas(self.mesh)
        self.ncells = mesh.topology.index_map(self.dim).size_local

        self.dx = ufl.Measure('dx',self.mesh)

        self.Vh = dolfinx.fem.functionspace(self.mesh,self.u_el)
        self.Qh = dolfinx.fem.functionspace(self.mesh,self.A_el)
        self.Th = dolfinx.fem.functionspace(self.mesh,self.tau_el)
        self.Wh = dolfinx.fem.functionspace(self.mesh,self.theta_el)

        if hasattr(self, 'Ah') and hasattr(self, 'thetah') and hasattr(self,'uhs'):
            # project functions into the new function spaces
            self.Ah = _interpolate(self.Qh, self.Ah, False)
            self.thetah = _interpolate(self.Wh, self.thetah, False)
            self.uhs = [_interpolate(self.Vh, uh, False) for uh in self.uhs]
        else:
            # create functions for the first time
            self.Ah = dolfinx.fem.Function(self.Qh)
            self.thetah = dolfinx.fem.Function(self.Wh)
            self.uhs = [dolfinx.fem.Function(self.Vh) for l in self.loads]

        self.tauhs = [dolfinx.fem.Function(self.Th) for l in self.loads]
        self.tau_exprs = [dolfinx.fem.Expression(self.Ah*_epsilon(uh,self.dim), self.Th.element.interpolation_points()) for uh in self.uhs]
        self.ah = dolfinx.fem.create_form(self.a, [self.Vh,self.Vh], self.mesh,{},{self.A:self.Ah},{})
        self._build_problems()

    def solve_all(self):
        compl = []

        for i,p in enumerate(self.problems):
            p.solve()
            self.tauhs[i].interpolate(self.tau_exprs[i])
            compl.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.Lh[i](self.uhs[i]))))

        return compl,dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.thetah*self.dx))
    
    def _build_problems(self):
        self.measures = []
        self.Lh = []
        self.problems = []
        self.bcsh = []

        for dl in self.dlocators:
            dirichlet_dof = dolfinx.fem.locate_dofs_geometrical(self.Vh, dl)
            self.bcsh.append(dolfinx.fem.dirichletbc(np.zeros(self.dim, ), dirichlet_dof, self.Vh))
            
        for vnl in self.vnlocators:
            facets = dolfinx.mesh.locate_entities_boundary(self.mesh, self.dim-1, vnl)
            facets_tag = dolfinx.mesh.meshtags(self.mesh,self.dim-1,facets,np.ones(len(facets),dtype=np.int32))
            self.measures.append(ufl.Measure('ds', domain=self.mesh, subdomain_data=facets_tag))

        v = ufl.TestFunction(self.Vh)
        for g,ds in zip(self.loads,self.measures):
            g_ = dolfinx.fem.Constant(self.mesh,g)
            self.Lh.append(ufl.inner(v,g_)*ds(1))

        for L,bc,uh in zip(self.Lh,self.bcsh,self.uhs):
            self.problems.append(
                petsc.LinearProblem(
                    self.ah, L, u=uh, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
                )
            )

    def compute_oc(self, FAinv, l):
        #Convert FAinv into a function
        FAinv_h = dolfinx.fem.Function(self.Qh) # same space as Ah
        FAinv_h.x.array[:] = FAinv.reshape(-1)
        oc_h = dolfinx.fem.Function(self.Wh)

        gtau= sum(ufl.inner(ufl.dot(FAinv_h,tauh), tauh) for tauh in self.tauhs)
        oc_expr = dolfinx.fem.Expression(ufl.min_value(1., ufl.sqrt(gtau/l)), self.Wh.element.interpolation_points())
        oc_h.interpolate(oc_expr)

        return np.dot(self.cell_areas, np.abs(oc_h.x.array-self.thetah.x.array))
        
    def compute_indicator(self, FAinv,l, delta, Ave, sigma=1.):
        ncells = self.mesh.topology.index_map(self.dim).size_local
        Wh = self.Wh #DG0 scalar, cell wise constant
        w = ufl.TestFunction(Wh)

        uhs = self.uhs
        pi_uhs = [local_quad_interpolation_cy(uh) for uh in uhs]
        tauhs_mat = [_svecT(tauh, self.dim) for tauh in self.tauhs]
        dx = ufl.Measure('dx', self.mesh)
        dss = self.measures
        dS = ufl.Measure('dS',self.mesh)
        n = ufl.FacetNormal(self.mesh)

        # error estimates
        eta = np.zeros((ncells,))

        # indicators in u
        for uh,pi_uh,tauh,ds,g in zip(uhs, pi_uhs, tauhs_mat, dss, self.loads):
            #compute weighting terms
            omega = ufl.inner(pi_uh-uh,pi_uh-uh)*w*ds(1)
            omega += ufl.inner(pi_uh('+')-uh,pi_uh('+')-uh)*w('+')*dS
            omega += ufl.inner(pi_uh('-')-uh,pi_uh('-')-uh)*w('-')*dS
            omega_np = np.sqrt(dolfinx.fem.assemble_vector(dolfinx.fem.form(omega)).array)

            #compute residuals
            rho = ufl.inner(ufl.as_vector(g)-ufl.dot(tauh,n),ufl.as_vector(g)-ufl.dot(tauh,n))*w*ds(1)
            rho += ufl.inner(tauh('+')*n('+')+tauh('-')*n('-'),tauh('+')*n('+')+tauh('-')*n('-'))*(w('+')+w('-'))*dS
            rho_np = np.sqrt(dolfinx.fem.assemble_vector(dolfinx.fem.form(rho)).array)

            eta += omega_np*rho_np
        
        # indicators in theta, first convert FAinv into a function
        FAinv_h = dolfinx.fem.Function(self.Qh) # same space as Ah
        FAinv_h.x.array[:] = FAinv.reshape(-1)

        # compute p_theta
        L = w*l*dx
        for pi_uh in pi_uhs:
            eps = _epsilon(pi_uh, self.dim)
            tau = ufl.dot(self.Ah, eps)
            L -= w*ufl.inner(ufl.dot(FAinv_h,tau),tau)/self.thetah**2*dx
        p_theta = riesz_representation(L)

        #compute theta_h_tilde
        theta_h_tilde = dolfinx.fem.Function(Wh)
        theta_h_tilde.x.array[:] = np.maximum(delta, np.minimum(1., self.thetah.x.array-sigma*p_theta.x.array))

        #weighting terms:
        omega_np = 0.5*np.abs(theta_h_tilde.x.array-self.thetah.x.array)
        rho = w*l*dx
        for tauh in self.tauhs:
            rho -= ufl.inner(ufl.dot(FAinv_h,tauh),tauh)/self.thetah**2*w*dx
        rho_np = np.abs(dolfinx.fem.assemble_vector(dolfinx.fem.form(rho)).array)

        eta += omega_np*rho_np

        return eta
