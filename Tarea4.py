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

# QR with shift for eigenvalues
def QR_shift(Matrix,shift,iterations):
    '''This function approximates the eigenvalues of a squared matrix by QR iterations and
    accelerates the process by adding a shift'''
    import numpy as np
    S = shift*np.identity( np.shape(Matrix)[0] )
    norm = np.max( np.abs([Matrix[0,1],Matrix[0,2],Matrix[1,0],Matrix[1,2],Matrix[2,0],Matrix[2,1]] ) )
    for i in range(iterations):
        Q, R = QR(Matrix - S)
        Matrix = R @ Q + S
        print(Matrix)
    return np.diag(Matrix)

if __name__ == "__main__":
    import numpy as np
    eigvals = []
    powers = [1,3,4,5]
    for N in powers:
        epsilon = 10**(-N)
        A = np.array( [ [8,1,0], [1,4,epsilon],[0,epsilon,1] ] )
        values = QR_shift(A,1,1000)
        eigvals.append(values)