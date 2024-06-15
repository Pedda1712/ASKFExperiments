"""
Sample code automatically generated on 2022-04-19 11:48:46

by geno from www.geno-project.org

from input

parameters
  matrix Kold symmetric
  scalar delta
  scalar c
  scalar gamma
  vector y
  vector eigenvaluesOld
  matrix eigenvectors
variables
  vector alphas
  vector eigenvalues
min
  0.5*(alphas.*y)'*eigenvectors*diag(eigenvalues)*eigenvectors'*(alphas.*y)-sum(alphas)+gamma*norm2(Kold-eigenvectors*diag(eigenvalues)*eigenvectors')
st
  alphas >= 0
  alphas <= c
  y'*alphas == 0
  eigenvalues >= 0
  sum(alphas) == 1
  sum(abs(eigenvalues)) <= delta*sum(abs(eigenvaluesOld))


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
import numpy
import sys
# from AuralSonarTests import AuralSonar_Globals

try:
    from ..genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver_binary not installed. Using SciPy solver_binary instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)



class GenoNLP:
    def __init__(self, Kold, delta, c, gamma, y, eigenvaluesOld, eigenvectors, np):
        self.np = np
        self.Kold = Kold
        self.delta = delta
        self.c = c
        self.gamma = gamma
        self.y = y
        self.eigenvaluesOld = eigenvaluesOld
        self.eigenvectors = eigenvectors
        assert isinstance(Kold, self.np.ndarray)
        dim = Kold.shape
        assert len(dim) == 2
        self.Kold_rows = dim[0]
        self.Kold_cols = dim[1]
        if isinstance(delta, self.np.ndarray):
            dim = delta.shape
            assert dim == (1, )
            self.delta = delta[0]
        self.delta_rows = 1
        self.delta_cols = 1
        if isinstance(c, self.np.ndarray):
            dim = c.shape
            assert dim == (1, )
            self.c = c[0]
        self.c_rows = 1
        self.c_cols = 1
        if isinstance(gamma, self.np.ndarray):
            dim = gamma.shape
            assert dim == (1, )
            self.gamma = gamma[0]
        self.gamma_rows = 1
        self.gamma_cols = 1
        assert isinstance(y, self.np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        assert isinstance(eigenvaluesOld, self.np.ndarray)
        dim = eigenvaluesOld.shape
        assert len(dim) == 1
        self.eigenvaluesOld_rows = dim[0]
        self.eigenvaluesOld_cols = 1
        assert isinstance(eigenvectors, self.np.ndarray)
        dim = eigenvectors.shape
        assert len(dim) == 2
        self.eigenvectors_rows = dim[0]
        self.eigenvectors_cols = dim[1]
        self.alphas_rows = self.Kold_rows
        self.alphas_cols = 1
        self.alphas_size = self.alphas_rows * self.alphas_cols
        self.eigenvalues_rows = self.eigenvectors_cols
        self.eigenvalues_cols = 1
        self.eigenvalues_size = self.eigenvalues_rows * self.eigenvalues_cols
        # the following dim assertions need to hold for this problem
        assert self.Kold_rows == self.eigenvectors_rows == self.alphas_rows == self.y_rows == self.Kold_cols
        assert self.eigenvalues_rows == self.eigenvectors_cols

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.alphas_size
        bounds += [0] * self.eigenvalues_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.alphas_size
        bounds += [inf] * self.eigenvalues_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.alphasInit = self.np.zeros((self.alphas_rows, self.alphas_cols))
        self.eigenvaluesInit = self.np.zeros((self.eigenvalues_rows, self.eigenvalues_cols))
        return self.np.hstack((self.alphasInit.reshape(-1), self.eigenvaluesInit.reshape(-1)))

    def variables(self, _x):
        alphas = _x[0 : 0 + self.alphas_size]
        eigenvalues = _x[0 + self.alphas_size : 0 + self.alphas_size + self.eigenvalues_size]
        return alphas, eigenvalues

    def fAndG(self, _x):
        alphas, eigenvalues = self.variables(_x)
        t_0 = (alphas * self.y)
        t_1 = (self.eigenvectors.T).dot(t_0)
        t_2 = (self.eigenvectors).dot((eigenvalues * t_1))
        T_3 = (self.Kold - (self.eigenvectors).dot((eigenvalues[:, self.np.newaxis] * self.eigenvectors.T)))
        f_ = (((0.5 * (t_0).dot(t_2)) - self.np.sum(alphas)) + (self.gamma * self.np.linalg.norm(T_3, 'fro')))
        g_0 = (((0.5 * (t_2 * self.y)) - self.np.ones(self.alphas_rows)) + (0.5 * ((self.eigenvectors).dot((t_1 * eigenvalues)) * self.y)))
        g_1 = ((0.5 * (t_1 * t_1)) - ((self.gamma / self.np.linalg.norm((self.Kold.T - ((self.eigenvectors * eigenvalues[self.np.newaxis, :])).dot(self.eigenvectors.T)), 'fro')) * self.np.diag(((self.eigenvectors.T).dot(T_3)).dot(self.eigenvectors))))
        g_ = self.np.hstack((g_0, g_1))
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        f = (self.y).dot(alphas)
        return f

    def gradientEqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        g_0 = (self.y)
        g_1 = (self.np.ones(self.eigenvalues_rows) * 0)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        alphas, eigenvalues = self.variables(_x)
        gv_0 = ((_v * self.y))
        gv_1 = (self.np.ones(self.eigenvalues_rows) * 0)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueEqConstraint001(self, _x):
        alphas, eigenvalues = self.variables(_x)
        f = (self.np.sum(alphas) - 1)
        return f

    def gradientEqConstraint001(self, _x):
        alphas, eigenvalues = self.variables(_x)
        g_0 = (self.np.ones(self.alphas_rows))
        g_1 = (self.np.ones(self.eigenvalues_rows) * 0)
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdEqConstraint001(self, _x, _v):
        alphas, eigenvalues = self.variables(_x)
        gv_0 = ((_v * self.np.ones(self.alphas_rows)))
        gv_1 = (self.np.ones(self.eigenvalues_rows) * 0)
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_

    def functionValueIneqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        f = (self.np.sum(self.np.abs(eigenvalues)) - (self.delta * self.np.sum(self.np.abs(self.eigenvaluesOld))))
        return f

    def gradientIneqConstraint000(self, _x):
        alphas, eigenvalues = self.variables(_x)
        g_0 = (self.np.ones(self.alphas_rows) * 0)
        g_1 = (self.np.sign(eigenvalues))
        g_ = self.np.hstack((g_0, g_1))
        return g_

    def jacProdIneqConstraint000(self, _x, _v):
        alphas, eigenvalues = self.variables(_x)
        gv_0 = (self.np.ones(self.alphas_rows) * 0)
        gv_1 = ((_v * self.np.sign(eigenvalues)))
        gv_ = self.np.hstack((gv_0, gv_1))
        return gv_


def solve(Kold, gamma,delta, c,  y, eigenvaluesOld, eigenvectors, np, max_iterations=3000):
    start = timer()
    NLP = GenoNLP(Kold, delta, c, gamma, y, eigenvaluesOld, eigenvectors, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver_binary options, they can be omitted.
    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : max_iterations,
               'm' : 10,
               'ls' : 0,
               'verbose' : 1  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver_binary version is sufficient.
        check_version('0.1.0')
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jacprod' : NLP.jacProdEqConstraint000},
                       {'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint001,
                        'jacprod' : NLP.jacProdEqConstraint001},
                       {'type' : 'ineq',
                        'fun' : NLP.functionValueIneqConstraint000,
                        'jacprod' : NLP.jacProdIneqConstraint000})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                      constraints=constraints, np=np)
    else:
        sys.exit('We do not want to use scipy - only genosolver!!')
        # SciPy: for inequality constraints need to change sign f(x) <= 0 -> f(x) >= 0
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jac' : NLP.gradientEqConstraint000},
                       {'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint001,
                        'jac' : NLP.gradientEqConstraint001},
                       {'type' : 'ineq',
                        'fun' : lambda x: -NLP.functionValueIneqConstraint000(x),
                        'jac' : lambda x: -NLP.gradientIneqConstraint000(x)})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    # assemble solution and map back to original problem
    alphas, eigenvalues = NLP.variables(result.x)
    elapsed = timer() - start
    print('solving took %.3f sec' % elapsed)
    return result, alphas, eigenvalues


def learn_spectral_properties(eigenvalues_old, eigenvectors, label, curr_run_settings, settings):

    K_old = eigenvectors @ (numpy.eye(len(eigenvalues_old)) * eigenvalues_old) @ eigenvectors.T;
    y = label

    gamma = curr_run_settings['gamma_param']
    delta = curr_run_settings['delta_param']
    c_param = curr_run_settings['c_param']

    result, alphas, eigenvalues_new = solve(K_old, delta, c_param, gamma, y, eigenvalues_old, eigenvectors, numpy, settings)

    K_new = eigenvectors @ numpy.eye(len(eigenvalues_new)) * eigenvalues_new @ eigenvectors.T
    label_new = numpy.copy(y)
    label_new[label_new == -1] = 0
    predictions = ((K_new @ (alphas * y)) >= 0).astype(int)
    accuracy = numpy.mean(predictions == label_new)

    return eigenvalues_new
