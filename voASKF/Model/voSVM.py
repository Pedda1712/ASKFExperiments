import numpy as np
from ..Solver import solve

def get_spectral_properties(kernel_matrices, subsample_size):
    m = len(kernel_matrices)
    n = len(kernel_matrices[0])
    if subsample_size == 'n_m':
        n_new = n * m
    elif subsample_size * n > n * m:
        n_new = n * m
    else:
        n_new = np.ceil(n * subsample_size).astype(int)

    all_eigenvalues = np.zeros(n)
    all_eigenvectors = np.zeros((n, n))
    all_eigenvalues_classes = np.zeros(n)
    for idx, kernel_matrix in enumerate(kernel_matrices):
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
        all_eigenvalues = np.append(all_eigenvalues, eigenvalues)
        all_eigenvectors = np.append(all_eigenvectors, eigenvectors, axis=1)
        all_eigenvalues_classes = np.append(all_eigenvalues_classes, idx * np.ones(n))

    all_eigenvalues = all_eigenvalues[n:]
    all_eigenvalues_classes = all_eigenvalues_classes[n:]
    all_eigenvectors = all_eigenvectors[:, n:]
    absolute_eigenvalues = np.abs(all_eigenvalues)

    ind = np.sort(np.argpartition(absolute_eigenvalues, -n_new)[-n_new:])

    selected_eigenvalues = all_eigenvalues[ind]
    selected_eigenvectors = all_eigenvectors[:, ind]
    selected_eigenvalues_classes = all_eigenvalues_classes[ind]

    # if True:
    #     for kernel_matrix in kernel_matrices:
    #         eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
    #         all_eigenvalues = np.append(all_eigenvalues, eigenvalues)
    #         all_eigenvectors = np.append(all_eigenvectors, eigenvectors, axis=1)
    #
    #     all_eigenvalues = all_eigenvalues[n:]
    #     all_eigenvectors = all_eigenvectors[:, n:]
    #     absolute_eigenvalues = np.abs(all_eigenvalues)
    #     ind = np.sort(np.argpartition(absolute_eigenvalues, -n_new)[-n_new:])
    #     all_eigenvalues = all_eigenvalues[ind]
    #     all_eigenvectors = all_eigenvectors[:, ind]
    #
    # else:
    #
    #     for kernel_matrix in kernel_matrices:
    #         eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
    #         all_eigenvalues = np.append(all_eigenvalues, eigenvalues)
    #         all_eigenvectors = np.append(all_eigenvectors, eigenvectors, axis=1)
    #         absolute_eigenvalues = np.abs(all_eigenvalues)
    #
    #         ind = np.sort(np.argpartition(absolute_eigenvalues, -n_new)[-n_new:])
    #
    #         all_eigenvalues = all_eigenvalues[ind]
    #         all_eigenvectors = all_eigenvectors[:, ind]

    return {'eigenvalues': selected_eigenvalues,
            'eigenvectors': selected_eigenvectors,
            'eigenvalue_orig_indices': selected_eigenvalues_classes,
            'eigenvalue_indices':ind}

# hypertet vertices
def mkVecLabel(cl, len):
        vec = np.zeros((len, 1))
        for i in range(0, len):
                if i == cl:
                        vec[i] = np.sqrt((len-1)/(len))
                else:
                        vec[i] = (-1)/np.sqrt(len * (len-1))
        return vec

class ASKFvoSVM:
        # K : data similarity matrix
        # labels : classification
        # max_iter: maximum inner iterations on solver
        def __init__(self, Ks, labels, max_iter = 1000, on_gpu=False):
                self.noLabels = np.max(labels)+1
                self.labels = labels

                # construct label vector matrix
                self.Y = None
                for l in np.nditer(self.labels):
                        vec = mkVecLabel(l, self.noLabels)
                        if self.Y is None:
                                self.Y = vec
                        else:
                                self.Y = np.append(self.Y, vec, axis=1)
                                
                # decompose kernel matrices
                eigenprops = get_spectral_properties(Ks, 1.0)
                self.old_eigenvalues = eigenprops['eigenvalues']
                self.eigenvectors = eigenprops['eigenvectors']
                self.eigenvalue_matrix_indices = eigenprops['eigenvalue_orig_indices']
                
                # old combined kernel space
                K_old = self.eigenvectors @ np.diag(self.old_eigenvalues) @ self.eigenvectors.T;
		        
                # solve ASKF problem
                print("start solving")
                self.beta = -5
                self.gamma = 10
                self.delta = 50
                self.C = 0.01
                self.Ky = self.Y.T @ self.Y

                if not on_gpu:
                    self.result, self.a , self.new_eigenvalues = solve(
                        Kold=K_old,
			beta=self.beta,
                        gamma=self.gamma,
                        delta=self.delta,
                        c=self.C,
                        Y=self.Y,
                        Ky = self.Ky,
                        eigenvaluesOld=self.old_eigenvalues,
                        eigenvectors=self.eigenvectors,
                        np=np,
                        max_iter=max_iter)
                else:
                    # move data to gpu and solve there
                    import cupy as cp
                    cKold = cp.asarray(K_old)
                    cY = cp.asarray(self.Y)
                    cKy = cp.asarray(self.Ky)
                    ceigenvaluesOld = cp.asarray(self.old_eigenvalues)
                    ceigenvectors = cp.asarray(self.eigenvectors)
                    cresult, ca, cnew_eigenvalues  = solve(
                        Kold=cKold,
                        gamma=self.gamma,
			beta = self.beta,
                        delta=self.delta,
                        c=self.C,
                        Y=cY,
                        Ky=cKy,
                        eigenvaluesOld=ceigenvaluesOld,
                        eigenvectors=ceigenvectors,
                        np=cp,
                        max_iter=max_iter)
                    self.result = cp.asnumpy(cresult)
                    self.a = cp.asnumpy(ca)
                    self.new_eigenvalues = cp.asnumpy(cnew_eigenvalues)

                self.K_new = self.eigenvectors @ np.diag(self.new_eigenvalues) @ self.eigenvectors.T
                
		        # get indices of SVs
                self.svinds = []
                self.n_samples = self.labels.shape[0]

                for ind in range(0, self.n_samples):
                        if self.a[ind] > 0.0:
                                self.svinds.append(ind)
                
                # create projection matrix from old space to new space
                K_sum = np.zeros(Ks[0].shape)
                for K in Ks:
                    K_sum += K
                    
                self.P = self.K_new @ np.linalg.pinv(K_sum)

                # bias vector
                self.b = -(self.Y) + (np.multiply(self.a, self.Y) @ self.K_new)
                # average the biases of all support vectors for robustness
                self.b = np.mean(self.b[:, self.svinds], axis=1)
        
        # ktest: similarity of test data to training data
        def predict(self, Ks_test):
                # project test similarities into learned space
                K_test_sum = np.zeros(Ks_test[0].shape)
                for K_test_orig in Ks_test:
                    K_test_sum += K_test_orig

                ktest = (self.P @ K_test_sum.T).T
                
                # vo dot product comparison
                scores = []
                for t in range(0, self.noLabels):
                        yt = mkVecLabel(t, self.noLabels)
                        ky = yt.T @ self.Y
                        sim = np.repeat(-(yt.T @ self.b), ktest.shape[0]).reshape(-1,1) + (ktest @ np.multiply(self.a, ky).T)
                        scores.append(sim)
                
                scores = np.hstack(scores)
                return (np.argmax(scores, axis=1))

