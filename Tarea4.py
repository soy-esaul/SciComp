# Function from a previous homework
def QR(A):
    '''Implementation of the Gram-Schmidt algorithm for QR factorization as seen on
    Trefethen and Bau "Numerical Linear Algebra" (1997).
    
    Arguments:
    - A: A full rank matrix (numpy array of shape (n,m))
    
    Returns:
    - Q: A matrix such that Q* Q = I and Q Q* = P, with P an orthogonal projector
    - R: An upper triangular matrix such that A = QR'''
    import numpy as np
    m, n = A.shape
    V = A.astype(float)
    Q = V.copy()
    R = np.zeros((n,n), dtype=float)
    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i] / R[i,i]
        for j in range(i+1,n):
            R[i,j] = np.matmul(Q[:,i].T,V[:,j])
            V[:,j] -= R[i,j]*Q[:,i]
    return Q, R
def QR_shift(A,s):
    '''This function finds the '''