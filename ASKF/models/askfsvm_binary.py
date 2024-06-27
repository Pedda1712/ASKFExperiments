from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..Solver import ASFSolver_original
from ..utils import MatrixDecomposition

from sklearn.metrics import accuracy_score

class ASKFSVMBinary(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=1000, subsample_size=1.0, beta=-1, gamma=1, delta=1, on_gpu=False):
        self.C = C  # Regularization parameter
        self.max_iter = max_iter  # Maximum number of iterations
        self.subsample_size = subsample_size  # Size of subsamples to use in training
        self.alphas = None  # Lagrange multipliers
        self.support_vectors = None  # Support vectors
        self.support_vectors_y = None  # Labels of the support vectors
        self.old_eigenvalues = None
        self.eigenvectors = None
        self.new_eigenvalues = None
        self.bias = 0  # Bias term
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.on_gpu = on_gpu

    def _convert_labels(self, y):
        # Convert labels to -1 and 1
        self.classes = np.unique(y)
        return np.where(y == self.classes[0], -1, 1)

    def fit(self, Ks, y):
        # self.y = self._convert_labels(y)
        self.classes = np.unique(y)
        self.y = y
        eigenprops = MatrixDecomposition.get_spectral_properties(Ks, self.subsample_size)

        self.old_eigenvalues = eigenprops['eigenvalues']
        self.eigenvectors = eigenprops['eigenvectors']
        self.eigenvalue_matrix_indices = eigenprops['eigenvalue_orig_indices']


        K_old = self.eigenvectors @ np.diag(self.old_eigenvalues) @ self.eigenvectors.T;

        # Solve the dual problem and compute 'alphas' and 'new_eigenvalues'
        if not self.on_gpu:
            result, alphas , new_eigenvalues = ASFSolver_original.solve(Kold=K_old, beta=self.beta, gamma=self.gamma, delta=self.delta, c=self.C, y=self.y,
                                     eigenvaluesOld=self.old_eigenvalues, eigenvectors=self.eigenvectors, np=np, max_iterations = self.max_iter)
        else:
            import cupy as cp
            cy = cp.asarray(self.y)
            ceigenvaluesOld = cp.asarray(self.old_eigenvalues)
            ceigenvectors = cp.asarray(self.eigenvectors)
            cKold = cp.asarray(K_old)
            cresult, calphas, cnew_eigenvalues = ASFSolver_original.solve(Kold=cKold, beta=self.beta, gamma=self.gamma, delta=self.delta, c=self.C, y = cy,
                                    eigenvaluesOld=ceigenvaluesOld, eigenvectors=ceigenvectors, np=cp, max_iterations = self.max_iter)
            result = cp.asnumpy(cresult)
            alphas = cp.asnumpy(calphas)
            new_eigenvalues = cp.asnumpy(cnew_eigenvalues)


        self.alphas = alphas
        self.new_eigenvalues = new_eigenvalues

        K_new = self.eigenvectors @ np.diag(self.new_eigenvalues) @ self.eigenvectors.T

        # CREATE FOR SOME TEST REASONS A PROJECTION MATRIX
        K_sum = np.zeros((len(self.y),len(self.y)))
        for K in Ks:
            K_sum += K

        self.projMatrix = np.dot(K_new,np.linalg.pinv(K_sum))

        # calculate bias term
        b_values = self.y - np.sum(self.alphas * self.y * K_new, axis=1)
        self.bias = np.mean(b_values)  # Average b over all support vectors for robustness

        y_predict = np.dot(self.alphas * self.y, K_new)# - self.bias
        y_pred = np.where(y_predict <= 0, self.classes[0], self.classes[1])

        mean_acc = accuracy_score(y_true=y,y_pred=y_pred)
        print("Accuracy: {}".format(mean_acc))

    def decision_function(self, Ks_test):

        #
        #   FOR SIMPLICITY WE USE THE PROJECTION INSTEAD OF THE OUT-OF-SAMPLE EXTENSION FROM
        #   https://arxiv.org/pdf/1411.1646.pdf
        #

        K_test_sum = np.zeros(Ks_test[0].shape)

        for K_test_orig in Ks_test:
            K_test_sum += K_test_orig

        K_test_proj = np.dot(self.projMatrix, K_test_sum.T)

        # THIS IS THE ALTERNATIVE SOLUTION FOR AN EFFICIENT OUT-OF-SAMPLE EXTENSION
        # all_eigenvectors = self.eigenvectors
        # all_eigenvalues = self.new_eigenvalues
        #
        # for idx, K_test_orig in enumerate(Ks_test):
        #     indices_m = np.where(self.eigenvalue_matrix_indices==idx)[0].tolist()
        #     eigenvectors_m = self.eigenvectors[:,indices_m]
        #     eigenvalues_m = self.new_eigenvalues[indices_m]
        #
        #     B = eigenvectors_m @ eigenvalues_m
        #     K_inner_m = eigenvectors_m @ np.diag(eigenvalues_m) @ eigenvectors_m.T
        #
        #
        #
        #     K_inner_m_pinv = np.linalg.pinv(eigenvectors_m @ np.diag(eigenvalues_m) @eigenvectors_m.T)
        #     K_test_m = K_test_orig.T @ K_inner_m_pinv @ K_test_orig
        #     K_test += K_test_m
        #     pause = "pause"

        y_predict = np.dot(self.alphas * self.y, K_test_proj) - self.bias


        return y_predict



    def predict(self, Ks_test):


        y_predict = self.decision_function(Ks_test)
        # Convert predictions back to original labels
        y_predict = np.where(y_predict <= 0, self.classes[0], self.classes[1])
        return y_predict
