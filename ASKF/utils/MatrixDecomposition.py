import numpy as np


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
