import numpy as np

def get_norm_std(weight):
    norms = np.linalg.norm(weight, axis=1)
    return np.std(norms) / np.mean(norms)

def get_mutual_coherence(weight):
    N, _ = np.shape(weight)
    norms = np.linalg.norm(weight, axis = 1)
    weight_normalized = weight / norms.reshape((N, 1))
    temp = weight_normalized @ weight_normalized.T
    temp += np.ones((N, N)) / (N-1)
    temp -= np.diag(np.diag(temp))
    return np.linalg.norm(temp, 1).item() / (N*(N-1))

