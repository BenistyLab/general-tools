import numpy as np
from scipy.linalg import eigh, svd, logm, expm, pinv
from scipy.sparse import eye
import warnings

np.random.seed(42)

def sym_pos_def_dist(A, B, p=2):
    eig = np.linalg.eigvals(np.linalg.inv(A) @ B)
    if p == 1:
        dist = np.sum(np.abs(np.log(eig)))
    else:
        dist = np.sum(np.abs(np.log(eig)) ** p) ** (1 / p)
    return dist

def sym_pos_semi_def_dist(A, B, r, k=1):
    sym = lambda M: (M + M.T) / 2
    A = sym(A)
    B = sym(B)

    eig_A, vec_A = np.linalg.eig(A)
    eig_B, vec_B = np.linalg.eig(B)

    # keep the eigenvectors of the r largest eigenvalues
    vec_A = vec_A[:, np.argsort(eig_A)[-r:]]
    vec_B = vec_B[:, np.argsort(eig_B)[-r:]]

    # numpy returns V transposed compared to matlab
    try:
        OA, S, OB = np.linalg.svd(vec_A.T @ vec_B)
    except:
        OA, S, OB = svd(vec_A.T @ vec_B, lapack_driver='gesvd')
    if np.any(abs(S) > 1.):
        if not np.allclose(abs(S[abs(S) > 1.]), 1.):
            print(f"SVD yields S {S[abs(S) > 1.]}")
        S[S > 1.] = 1.
        S[S < -1.] = -1.
    vTheta = np.arccos(S)
    UA = vec_A @ OA
    UB = vec_B @ OB.T
    RA = sym(UA.T @ A @ UA)
    RB = sym(UB.T @ B @ UB)
    dU = np.linalg.norm(vTheta)
    dR = sym_pos_def_dist(RA, RB)
    d = np.sqrt(dU ** 2 + k * dR ** 2)
    return d

def _riemannian_dist(corrs, eigval_bound=0.01):
    # r: smallest rank
    r = np.min(np.sum(np.linalg.eigvals(corrs) > eigval_bound, axis=1))

    dR = np.zeros((len(corrs), len(corrs)))
    for i, corr_i in enumerate(corrs):
        for j, corr_j in enumerate(corrs[i + 1:]):
            dR[i + j + 1, i] = sym_pos_semi_def_dist(corr_i, corr_j, r)
            dR[i, i + j + 1] = dR[i + j + 1, i]
    return dR

def safe_corr(x, y):
    # Ensure inputs are numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    # Check for constant vectors (std = 0) and prevent division by zero
    if np.std(x) == 0 or np.std(y) == 0:
        return 0  # Correlation is undefined, return 0 or NaN

    # Compute correlation
    corr_matrix = np.corrcoef(x, y)

    return corr_matrix[0, 1]

def get_corr_matrix(matrix):
    num_traces = len(matrix)
    corr_matrix = np.zeros((num_traces, num_traces))
    for i in range(num_traces):
        for j in range(num_traces):
            if i == j:
                corr_matrix[i, j] = 1
            else:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Catch all warnings
                    corr_value = np.corrcoef(matrix[i], matrix[j])[0, 1]

                    if w:  # If a warning was caught, use safe_corr instead
                        corr_value = safe_corr(matrix[i], matrix[j])

                corr_matrix[i, j] = corr_value
    return corr_matrix

def matrix_power_adj(A, p):
    eigvals, eigvecs = np.linalg.eigh(A)

    # Set a threshold for small negative eigenvalues (e.g., 1e-20)
    threshold = 0
    assert np.min(eigvals) > -1e-20, f"Overflow Detected, found negative eigenvalue {np.min(eigvals)}"
    eigvals = np.maximum(eigvals, threshold)  # Replace negative eigenvalues with zero
    # Raise the eigenvalues to the power of p
    eigvals_p = eigvals ** p  # Eigenvalues raised to the power of p
    # Reconstruct the matrix A^p
    A_inv_p = eigvecs @ np.diag(eigvals_p) @ eigvecs.T  # A^p

    return A_inv_p

def clip_eigenvalues(matrix, threshold1=1e5, threshold2=1e-5):
    """
    Clip the eigenvalues of a matrix to a specified threshold.

    Parameters:
    - matrix: 2D numpy array, the matrix whose eigenvalues are to be clipped.
    - threshold: The maximum value to clip eigenvalues to (default is 1e6).

    Returns:
    - clipped_matrix: 2D numpy array, the matrix with clipped eigenvalues.
    """
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Clip eigenvalues that are above the threshold
    clipped_eigenvalues = np.clip(eigenvalues, threshold2, threshold1)

    # Reconstruct the matrix with clipped eigenvalues
    clipped_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ np.linalg.inv(eigenvectors)

    return clipped_matrix

def fixed_geodes_eff(A, B, p):
    """
    Computes the point t along the geodesic stretching from A to B.

    Parameters:
        A (ndarray): PSD matrix of rank `dim`.
        B (ndarray): PSD matrix of rank `dim`.
        p (float): Desired point along the geodesic (t > 0).
    Returns:
        ndarray: The point t along the geodesic.
    """
    # Compute the ranks of A and B based on non-zero eigenvalues
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    dim = min(rank_A, rank_B)  # Set dim to the minimum rank
    if dim == np.shape(B)[0] and dim == np.shape(A)[0]:
        return clip_eigenvalues(np.real(filter(A, B, p)))
    # Get the largest `dim` eigenvalues and eigenvectors for A and B
    S1, U1 = np.linalg.eig(A)
    S2, U2 = np.linalg.eig(B)
    S1 = np.real(S1)  # Take only real parts of eigenvalues
    S2 = np.real(S2)

    U1 = U1[:, np.argsort(-S1)[:dim]]
    U2 = U2[:, np.argsort(-S2)[:dim]]
    S1 = -np.sort(-S1)[:dim]
    S2 = -np.sort(-S2)[:dim]

    # Extract eigenvector subspaces
    VA = np.real(U1[:, :dim])  # Ensure eigenvectors are real
    VB = np.real(U2[:, :dim])

    # Singular Value Decomposition (SVD)
    OA, SAB, OB = svd(VA.T @ VB)
    SAB = np.real(SAB)  # Ensure singular values are real

    UA = VA @ OA
    UB = VB @ OB.T
    theta = np.arccos(np.clip(SAB, -1, 1))  # Clip to avoid numerical errors

    # Compute intermediate matrices
    tmp = UB @ pinv(np.diag(np.sin(theta)))
    X = (eye(A.shape[0]).toarray() @ tmp - UA @ UA.T @ tmp)
    U = UA @ np.diag(np.cos(theta * p)) + X @ np.diag(np.sin(theta * p))

    # Compute R2
    RB2 = OB @ np.diag(S2) @ OB.T
    assert np.all(S1 > 0), "Not all eigenvalues are positive!"
    RA = OA.T @ np.diag(np.sqrt(S1)) @ OA
    RAm1 = OA.T @ np.diag(1 / np.sqrt(S1)) @ OA
    eigenvalues = np.real(np.linalg.eigvals(RAm1 @ RB2 @ RAm1))  # Ensure eigenvalues are real
    assert np.all(eigenvalues >= 0), "Not all eigenvalues are positive!"
    R2 = RA @ expm(p * logm(RAm1 @ RB2 @ RAm1)) @ RA
    # Compute the result
    S = U @ R2 @ U.T
    return clip_eigenvalues(np.real(S))  # Ensure final result is real




















