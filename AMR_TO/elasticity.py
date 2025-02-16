import ufl
from basix.ufl import element
import dolfinx
from dolfinx.fem import petsc

from mpi4py import MPI
import numpy as np

from typing import Optional

def _epsilon(v : any ,dim : int):
    """
    Strain in engineering notation
    """
    e = ufl.sym(ufl.grad(v))
    if dim == 2:
        return ufl.as_vector([e[0,0],e[1,1],np.sqrt(2)*e[0,1]])
    elif dim == 3:
        return ufl.as_vector([e[0,0],e[1,1],e[2,2],np.sqrt(2)*e[0,1],np.sqrt(2)*e[0,2],np.sqrt(2)*e[1,2]])
    
class _AbstractElasticityForm():
    """
    This class implements the abstract bilinear form int_Omega A e(u):e(v) d x used in the linear elasticity    
    """
    def __init__(self, dim : int):
        self.dim = dim # space dimension
        self.edim = 6 if dim == 3 else 3 # dimension of engineering vector
        
        # finite elements
        self.u_el = element('Lagrange','triangle',1,shape=(self.dim,)) # FE displacement
        self.A_el = element('Lagrange','triangle',0,shape=(self.edim,self.edim),discontinuous=True) #FE Hooke's law
        self.tau_el = element('Lagrange','triangle',0,shape=(self.edim,),discontinuous=True) #FE stress
        self.theta_el = element('Lagrange','triangle',0,discontinuous=True) #FE density

        #abstract mesh representation
        self.mesh_abs = ufl.Mesh(self.u_el)
        dx = ufl.Measure('dx', self.mesh_abs)

        #abstract function spaces
        self.V = ufl.FunctionSpace(self.mesh_abs,self.u_el)
        self.Q = ufl.FunctionSpace(self.mesh_abs,self.A_el)

        #abstract form
        self.u,self.v = ufl.TestFunction(self.V),ufl.TrialFunction(self.V)
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

    def update_mesh(self, mesh : dolfinx.mesh.Mesh, keep : Optional[bool] = False):
        """
        Update or assign for the first time the computation mesh        
        """
        if keep:
            raise NotImplementedError

        self.mesh = mesh

        self.dx = ufl.Measure('dx',self.mesh)

        self.Vh = dolfinx.fem.functionspace(self.mesh,self.u_el)
        self.Qh = dolfinx.fem.functionspace(self.mesh,self.A_el)
        self.Th = dolfinx.fem.functionspace(self.mesh,self.tau_el)
        self.Wh = dolfinx.fem.functionspace(self.mesh,self.theta_el)

        self.Ah = dolfinx.fem.Function(self.Qh)
        self.thetah = dolfinx.fem.Function(self.Wh)

        self.ah = dolfinx.fem.create_form(self.a, [self.Vh,self.Vh], self.mesh,{},{self.A:self.Ah},{})
        self.uhs = [dolfinx.fem.Function(self.Vh) for l in self.loads]
        self.tauhs = [dolfinx.fem.Function(self.Th) for l in self.loads]

        self.tau_exprs = [dolfinx.fem.Expression(self.Ah*_epsilon(uh,self.dim), self.Th.element.interpolation_points()) for uh in self.uhs]

        self._build_problems()

    def solve_all(self):
        compl = []

        for i,p in enumerate(self.problems):
            p.solve()
            self.tauhs[i].interpolate(self.tau_exprs[i])
            compl.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.Lh[i](self.uhs[i]))))

        return compl,dolfinx.fem.assemble_scalar(dolfinx.fem.form(self.thetah*self.dx))
    
    def _build_problems(self):
        measures = []

        self.Lh = []
        self.problems = []
        self.bcsh = []

        for dl in self.dlocators:
            dirichlet_dof = dolfinx.fem.locate_dofs_geometrical(self.Vh, dl)
            self.bcsh.append(dolfinx.fem.dirichletbc(np.zeros(self.dim, ), dirichlet_dof, self.Vh))
            
        for vnl in self.vnlocators:
            facets = dolfinx.mesh.locate_entities_boundary(self.mesh, self.dim-1, vnl)
            facets_tag = dolfinx.mesh.meshtags(self.mesh,self.dim-1,facets,np.ones(len(facets),dtype=np.int32))
            measures.append(ufl.Measure('ds', domain=self.mesh, subdomain_data=facets_tag))

        v = ufl.TestFunction(self.Vh)
        for g,ds in zip(self.loads,measures):
            g_ = dolfinx.fem.Constant(self.mesh,g)
            self.Lh.append(ufl.inner(v,g_)*ds(1))

        for L,bc,uh in zip(self.Lh,self.bcsh,self.uhs):
            self.problems.append(
                petsc.LinearProblem(
                    self.ah, L, u=uh, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
                )
            )
        