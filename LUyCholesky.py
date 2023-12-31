# Implementation of LU and Cholesky decomposition algorithms in Python


# Backward substitution
def backsubs(U,v):
    '''Implementation of the Backward substitution algorithm for linear systems solution
    as shown in "Trefethen and Bau, Numerical linear algebra. (1997)"
    
    Arguments:
    - U: An upper triangular matrix (numpy array of shape (n,n))
    - v: A vector of size n (numpy array of shape (n,1))
    
    Returns:
    - A vector x (numpy array of shape (n,1)) of values that solve for the system:
    Ux' = v

    In practice, the matrix needs no strictly to be triangular, but the function will
    ignore the above diagonal elements. This allows the function to be used on the 
    matrices coming from the LUP or Cholesky factorization algorithms.

    Requires:
    import numpy as np
    '''
    import numpy as np
    assert 0 not in np.diag(U), "Error: La matriz no es compatible"
    assert U.shape[0] == U.shape[1], "Error: La matriz no es cuadrada"
    assert U.shape[0] == v.shape[0], "Error: las dimensiones del sistema son incompatibles"
    x = v.astype(float)
    n = len(v)
    x[n-1] = v[n-1] / U[n-1,n-1]
    for i in range(len(v)-2, -1, -1):
        x[i] = ( v[i] - np.sum( np.multiply(x[i+1:],U[i,i+1:])) ) / U[i,i]
    return x

# Forward substitution
def forsubs(L,v):
    '''Implementation of the Forward substitution algorithm for linear systems solution
    as shown in "Trefethen and Bau, Numerical linear algebra. (1997)"
    
    Arguments:
    - L: A lower triangular matrix (numpy array of shape (n,n))
    - v: A vector of size n (numpy array of shape (n,1))
    
    Returns:
    - A vector x (numpy array of shape (n,1)) of values that solve for the system:
    Lx' = v

    In practice, the matrix needs no strictly to be triangular, but the function will
    ignore the below diagonal elements. This allows the function to be used on the 
    matrices coming from the LUP or Cholesky factorization algorithms.

    Requires:
    import numpy as np
    '''
    import numpy as np
    assert 0 not in np.diag(L), "Error: La matriz no es compatible"
    assert L.shape[0] == L.shape[1], "Error: La matriz no es cuadrada"
    assert L.shape[0] == v.shape[0], "Error: las dimensiones del sistema son incompatibles"
    x = v.astype(float)
    n = len(v)
    x[0] = v[0] / L[0,0]
    for i in range(len(v)-1):
        x[i+1] = ( v[i+1] - np.sum(np.multiply(x[:(i+1)],L[(i+1),:(i+1)])) ) / L[i+1,i+1]
    return x

# LUP
def LUP(A, out='copy'):
    '''Implementation of the LU decomposition algorithm with partial pivoting
    as shown in "Trefethen and Bau, Numerical linear algebra. (1997)"
    
    Arguments:
    - A: A squared matrix (numpy array of shape (n,n))
    
    Returns:
    A tuple (L,U,P) with:
    - L: A lower triangular matrix (numpy array of shape (n,n))
    - U: An upper triangular matrix (numpy array of shape (n,n))
    - P: A permutation matrix

    The matrices satisfy PA = LU when setting the below diagonal terms of U and
    the above diagonal terms of L equal to 0. In practice these entries are not 0 so that
    they can be used more efficiently and even saved inside a unique numpy array instead
    of two separated instances

    Requires:
    import numpy as np
    '''
    import numpy as np
    # Evaluate if the given matrix is squared
    assert A.shape[0] == A.shape[1], "Error: Matrix must be squared"
    assert out in {"compact","copy"}, "Error: Out value must be either 'compact', 'double' or 'copy'"
    # Modify the original matrix or copy it
    if out == "compact":
        n = A.shape[0]
        U = A.astype(float)
        P = np.identity(n)
        for k in range(n-1):
            i = np.argmax(np.abs(U[k:,k])) + k 
            t1 = U[k,k:].copy()
            t2 = U[i,k:].copy()
            U[i,k:] = t1
            U[k,k:] = t2
            t1 = U[i,:k].copy() 
            t2 = U[k,:k].copy()
            U[k,:k] = t1
            U[i,:k] = t2
            P[[i,k]] = P[[k,i]]
            for j in range(k+1,n):
                U[j,k] = U[j,k] / U[k,k]
                U[j,(k+1):n] -= U[j,k]*U[k,(k+1):n]
        return U, P
    elif out == "copy":
        n = A.shape[0]
        U = A.astype(float)
        L = np.identity(n)
        P = np.identity(n)
        for k in range(n-1):
            i = np.argmax(np.abs(U[k:,k])) + k 
            t1 = U[k,k:].copy()
            t2 = U[i,k:].copy()
            U[i,k:] = t1
            U[k,k:] = t2
            t1 = L[i,:k].copy() 
            t2 = L[k,:k].copy()
            L[k,:k] = t1
            L[i,:k] = t2
            P[[i,k]] = P[[k,i]]
            for j in range(k+1,n):
                L[j,k] = U[j,k] / U[k,k]
                U[j,(k+1):n] -= L[j,k]*U[k,(k+1):n]
        return L, U, P


# Cholesky
def cholesky(A,copy=True):
    '''Implementation of the Cholesky decomposition algorithm
    as shown in "Trefethen and Bau, Numerical linear algebra. (1997)"
    
    Arguments:
    - A: A hermitian positive definite squared matrix (numpy array of shape (n,n))
    
    Returns:
    - R: A matrix which above diagonal entries correspond to the Cholesky factorization 
    of A such that 

    R^T R = A

    Requires:
    import numpy as np
    '''
    import numpy as np
    n = A.shape[0]
    assert n == A.shape[1], "Error: Matrix must be squared!"
    assert not (0 in np.diag(A)), "Error: Matrix not compatible"
    if copy:
        R = A.astype(float)
        for k in range(n):
            for j in range(k+1,n):
                R[j,j:n] -= (R[k,j:n]*(R[k,j]/R[k,k]))
            R[k,k:n] = R[k,k:n] / np.sqrt( R[k,k] )
        return R
    else:
        A = A.astype(float)
        for k in range(n):
            for j in range(k+1,n):
                A[j,j:n] -= A[k,j:n]*(A[k,j]/A[k,k])
            A[k,k:n] = A[k,k:n] / np.sqrt( A[k,k] )
        return A

# Examples of use
if __name__ == "__main__":
    import numpy as np

    # Example of forward and backward substitution.
    # Note matrices are not necessarily diagonal because the function
    # will only use the above or below diagonal elements.
    U = np.matrix('1,2,3; 2,1,2; 3,4,1', dtype=float)
    L = np.matrix('1,3,4; 1,2,5; 1,2,3', dtype=float)
    v = np.array([1,5,0], dtype=float)

    z = forsubs(L,v)
    y = backsubs(U,v)

    print("La solución a Lz = v es ", z, "\n",
          "La solución a Uy = v es ", y)
    
    # Example of LUP decomposition
    E = np.matrix('2,1,1,0; 4,3,3,1; 8,7,9,5; 6,7,9,8',dtype=float)
    I, S, M = LUP(E, out="copy")

    print("L:", I, "\n U:", S, "\n", "P:", M)

    # Example of Cholesky
    C = np.matrix('6,15,55;15,55,225;55,225,979',dtype=float)
    C = cholesky(C,copy=False)
    
    print(C)