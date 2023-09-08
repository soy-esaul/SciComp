# Implementation of QR algorithm
# %%
def QR(A):
    '''Implementation of the Gram-Schmidt algorithm for QR factorization as seen on
    Trefethen and Bau "Numerical Linear Algebra" (1997)'''
    import numpy as np
    m = A.shape[0]
    n = A.shape[1]
    V = A.astype(float)
    Q = V.copy()
    R = np.zeros( (n,n) )
    for i in range(n):
        R[i,i] = np.linalg.norm(V[:,i])
        Q[:,i] = V[:,i] / R[i,i]
        for j in range(i+1,n):
            R[i,j] = Q[:,i].T @ V[:,j]
            V[:,j] -= R[i,j]*Q[:,i]
    return Q, R

def LSQR(X,b):
    '''Implementation of the Least Squares Algorithm using QR descomposition as seen on
    Trefethen and Bau "Numerical Linear Algebra" (1997)'''
    Q, R = QR(X)
    from LUyCholesky import backsubs
    y = np.matmul(Q.T , b)
    x = backsubs(R,y.T)
    return x
# %%

if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    A = stats.uniform.rvs(size=12)
    A = np.reshape(A,(4,3))
    Q, R = QR(A)
    print(Q,"\n",R)

    X = np.matrix("1,2,3,4;5,6,7,8;9,8,8,6;5,4,3,2;1,3,4,5", dtype=float)
    b = np.array([1,2,3,4,5], dtype=float)
    x = LSQR(X,b)
    print(x)
# %%
