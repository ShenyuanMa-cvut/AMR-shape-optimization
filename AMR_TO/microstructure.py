import sympy as sp
from sympy.polys.monomials import itermonomials,monomial_count
from sympy.polys.orderings import monomial_key

import numpy as np
import cvxpy as cvx

from functools import reduce
from itertools import product

class Microstructure():
    """
        Implementation of the optima SDP solution of finte rank laminates that minimizes the complementary energy
    """
    def __init__(self,d,mu,lmbda,q):
        self.d = d
        self.engineer_dim = 6 if d==3 else 3
        self.mu = mu
        self.lmbda = lmbda
        self.q = q
        self.A = _Hooke(self.mu, self.lmbda, self.d)

        #SYMPY objects
        self._v = sp.symbols(f'v:{self.d}') # symbolic variable in the sphere
        self._vvec = sp.Matrix(self._v) # column vector of symbolic variables
        self._basis = sorted(itermonomials(self._v, 4), key=monomial_key('grlex', self._v[::-1]))[monomial_count(d,3):] #basis vector 
        
        #CVXPY objects
        self._c = cvx.Variable(shape=(q,),name='c') #complementary energies
        self._y = cvx.Variable(shape=(len(self._basis,)),name='y') #fourth order moments
        self._tau = [cvx.Parameter(shape=(self.engineer_dim, ), name=f'tau{j}') for j in range(q)] #stress, in engineering notation
        self._weights = cvx.Parameter(shape=(q,), name='w') #manual normalization of the objective

        self._obj = self._weights@self._c #objective function
        self._build_moment_mat_cvx()
        self._build_finite_rank_laminates()
        self._build_cons()

        self._prob = cvx.Problem(cvx.Minimize(self._obj), self._cons)

    def _build_moment_mat_cvx(self):
        moment_mat = _moment_matrix(self.d, self._v)
        moment_mat_coef = _poly_matrix_to_coeffs(moment_mat, self._v, self._basis)
        self._My = sum((m_*y_ for m_,y_ in zip(moment_mat_coef, self._y)))

    def _build_finite_rank_laminates(self):
        f_Ac_expr = _f_Ac(self._v, self.mu, self.lmbda)
        f_Ac_expr_coef = _poly_matrix_to_coeffs(f_Ac_expr, self._v, self._basis)
        self._Fy = sum((f_*y_ for f_,y_ in zip(f_Ac_expr_coef, self._y)))+self.A

    def _build_cons(self):
        self._cons = []
        self._cons.append(self._My >> 0)
        self._cons.append(self._y @ _probability(self._v, self._basis) == 1)
        for i in range(self.q):
            self._cons.append(cvx.bmat([[self._c[i].reshape((1,1)),self._tau[i].reshape((1,self.engineer_dim))],[self._tau[i].reshape((self.engineer_dim,1)), self._Fy]]) >> 0)

    def compute_microstructure(self, tau : list[np.ndarray], **kwargs):
        """
            Compute the optimal complementary energies g(tau) and finite rank hookes law F_Ac
        """
        w = np.zeros(self.q)

        for i in range(self.q):
            ni = np.linalg.norm(tau[i])
            self._tau[i].value = tau[i]/ni
            w[i] = ni**2

        self._weights.value = w
        sol = self._prob.solve(**kwargs)
        return sol,self._Fy.value,self._My.value

def _right_mul(v : sp.Matrix):
    """
    Return the matrix representation R(v) of the right multiplication A |-> Av. 
    If A is a symmetric matrix and svec(A) its symmetric vectorization then R(v)svec(A) = Av
    v must be dimension 2 or 3
    """
    if v.shape[0] == 2:
        return sp.Matrix([[v[0],0,v[1]/sp.sqrt(2)],[0,v[1],v[0]/sp.sqrt(2)]])

    elif v.shape[0] == 3:
        return sp.Matrix([[v[0],0,0,v[1]/sp.sqrt(2),v[2]/sp.sqrt(2),0],
                          [0,v[1],0,v[0]/sp.sqrt(2),0,v[2]/sp.sqrt(2)],
                          [0,0,v[2],0,v[0]/sp.sqrt(2),v[1]/sp.sqrt(2)]])

def _tr_vec(d):
    """
    Obtain the vector repr of trace(A) = tr_vec.T svec(A)
    """
    if d==2:
        return sp.Matrix([[1],[1],[0]])
    elif d==3:
        return sp.Matrix([[1],[1],[1],[0],[0],[0]])


def _f_Ac(V : list[sp.Symbol],mu,lmbda):
    """
    Compute the homogeneous quartic part of the finite rank laminates formula
    """
    d = len(V)
    v = sp.Matrix([[V[i]] for i in range(d)])
    norm_squared = sum((v_**2 for v_ in V))
    T = _tr_vec(d)
    Rv = _right_mul(v)
    
    A = 4*mu**2*Rv.T*Rv+lmbda**2*T*T.T*norm_squared+2*mu*lmbda*(Rv.T*v*T.T+T*v.T*Rv) # |Axi v|^2
    A *= -norm_squared/mu

    B = 4*mu**2*Rv.T*v*v.T*Rv+2*mu*lmbda*norm_squared*(T*v.T*Rv+Rv.T*v*T.T)+lmbda**2*norm_squared**2*T*T.T
    B *= (lmbda+mu)/((lmbda+2*mu)*mu)
    
    return A + B

def _degree_list_to_monomial(dl,V):
    return reduce(lambda x,y : x*y, (v_**d_ for (d_,v_) in zip(dl,V)))

def _poly_matrix_to_coeffs(f,V,mbasis):
    """
    Convert an expression f into a list of matrix coefficients. 
    f is a matrx where each coefficient is a polynomial in some variable (x_1,...x_n).
    The goal of this function is to return a list of matrices [f_alpha,...] so 
    f(x) = sum_alpha f_alpha x^alpha
    the list of x^alpha is provided in the mbasis
    """
    
    rn = len(mbasis)
    mat_coef = [np.zeros(f.shape) for i in range(rn)]
    flist = f.tolist()
    
    for i, row in enumerate(flist):
        for j, col in enumerate(row):
            fij = sp.Poly(col,V)
            for degree_list,coeff in fij.terms():
                mon = _degree_list_to_monomial(degree_list, V)
                ind = mbasis.index(mon)

                mat_coef[ind][i,j] = float(coeff)
    
    return mat_coef

def _probability(V,mbasis):
    norm_4 = sp.Poly(sum((v_**2 for v_ in V))**2,V)
    p = np.zeros(len(mbasis))
    for dl,coef in norm_4.terms():
        mono = _degree_list_to_monomial(dl,V)
        ind = mbasis.index(mono)
        p[ind] = float(coef)
    return p

def _Hooke(mu,lmbda,d):
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

def _vec_index(d):
    if d == 2:
        return {(0,0):0,(0,1):2,(1,1):1}
    elif d == 3:
        return {
            (0,0):0,
            (0,1):3,
            (0,2):4,
            (1,1):1,
            (1,2):5,
            (2,2):2
        }

def _moment_matrix(d,V):
    engineer_dim = 6 if d==3 else 3
    Gram = sp.Matrix([[0 for j in range(engineer_dim)] for i in range(engineer_dim)])
    vec_ind = _vec_index(d)
    for (i,j,k,l) in product(*((i_ for i_ in range(d)) for _ in range(4))):
        alpha = vec_ind[(min(i,j),max(i,j))]
        beta = vec_ind[(min(k,l),max(k,l))]
        Gram[alpha,beta] += V[i]*V[j]*V[k]*V[l]
    return Gram