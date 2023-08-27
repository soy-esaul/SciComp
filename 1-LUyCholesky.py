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

    Requires:
    import numpy as np
    '''
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

    Requires:
    import numpy as np
    '''
    assert 0 not in np.diag(L), "Error: La matriz no es compatible"
    assert L.shape[0] == L.shape[1], "Error: La matriz no es cuadrada"
    assert L.shape[0] == v.shape[0], "Error: las dimensiones del sistema son incompatibles"
    x = v.astype(float)
    n = len(v)
    x[0] = v[0] / L[0,0]
    for i in range(len(v)-1):
        x[i+1] = ( v[i+1] - np.sum(np.multiply(x,L[i,:])) ) / L[i+1,i+1]
    return x





# Examples of use
if __name__ == "__main__":
    import numpy as np
    U = np.matrix('1,2,3; 0,1,2; 0,0,1')
    L = np.matrix('1,0,0; 1,2,0; 1,2,3')
    v = np.array([1,5,0])

    x = forsubs(L,v)
    y = backsubs(U,v)

    print("La solución a Lx = v es ",x, "\n",
          "La solución a Uy = v es ", y)