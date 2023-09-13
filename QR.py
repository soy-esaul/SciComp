# Implementation of QR algorithm
# %%
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

def LSQR(X,b):
    '''Implementation of the Least Squares Algorithm using QR descomposition as seen on
    Trefethen and Bau "Numerical Linear Algebra" (1997).
    
    Arguments:
    - X: A full rank matrix (numpy array of shape (n,m))
    - b: An m-dimensional vector (numpy array of shape(m,1))

    Returns:
    - x: A vector which is the least square solution to the problem min(|Ax - b|)
    '''
    import numpy as np
    Q, R = QR(X)
    from LUyCholesky import backsubs
    y = np.matmul(Q.T, b)
    x = backsubs(R, y.T)
    return x

def polfit(x,y,d,get_matrix=False):
    '''A function that fits a polynomial of given degree between two sets of points
    
    Arguments:
    - x: A vector of inputs (numpy array of shape (m,1) )
    - y: A vector of outputs (numpy array of shape (m,1) )
    - d: degree of the polynomial to be fitted with d < m
    - get_matrix: If True, the Vandermonde matrix will be returned
    
    Returns:
    - c: A vector of coefficients that is the solution to the least squares problem
        X c = y,
    where X is the Vandermonde matrix of x with degree d
    - X: Vandermonde matrix of x of degree d, returned if the option get_matrix is set to be True
    '''
    import numpy as np
    m = x.shape[0]
    assert d < m, "¡Error! El grado del polinomio debe ser menor que la dimensión del vector"
    X = np.ones((m,d+1), dtype=float)
    for i in range(1,d+1):
        X[:,i] = x**(i)
    c = LSQR(X,y)
    if get_matrix:
        return c, X
    else:
        return c
# %%
if __name__ == "__main__":
    # Import modules used
    import numpy as np
    from scipy.stats import norm
    from scipy.linalg import qr
    import matplotlib.pyplot as plt
    import matplotlib
    import time
    # Plot style
    matplotlib.style.use("seaborn-v0_8")
    qtimes = []
    stimes = []
    # Loop over degree of polynomials
    for n in [100,1000,10000]:
        # Simulate random noise
        epsilon = norm.rvs(loc=0,scale=0.11,size=n)
        # Regresion points
        x = np.array([ (4*np.pi*(i+1))/n for i in range(n) ], dtype=float)
        # Response variable
        y = np.sin(x) + epsilon
        # To plot the polynomal
        axis = np.linspace(np.min(x),np.max(x),num=n)
        # Loop over size of observations
        for p in [3,4,6,100]:
            # Get the coefficients and Vandermonde matrix with the previously 
            # defined function
            d = p-1
            coef, V = polfit(x,y,d,get_matrix=True)
            # Compare QR times
            start = time.time()
            Q, R = QR(V)
            end = time.time()
            qtimes.append(end-start)
            start = time.time()
            q, r = qr(V,mode="economic")
            end = time.time()
            stimes.append(end-start)
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(7,4))
            # To plot the polynomial
            polynomial = np.zeros(axis.shape,dtype=float)
            # Add the polynomial values
            for i in range(p):
                polynomial += coef[i]*(axis**i)
            # Scatterplot of the data points
            ax.scatter(x,y,marker=".",alpha=0.5,c="C0")
            # Polynomial plot
            ax.plot(axis,polynomial,"--",c="g")
            fig.suptitle("Gráfica para " + f"%i" %d + " grados, y " + f"%i" % n + " puntos")
    # Compare QR vs scipy QR run times
    fig2, ax2 = plt.subplots(figsize=(7,4))
    cases = [ "(3,100)","(4,100)","(6,100)","(100,100)","(3,1000)","(4,1000)","(6,1000)","(100,1000)","(3,10000)","(4,10000)","(6,10000)","(100,10000)"]
    ax2.scatter(cases,qtimes,label="Algoritmo propio",marker=",",alpha=0.5)
    ax2.scatter(cases,stimes,label="Scipy",marker=",",alpha=0.5)
    fig2.autofmt_xdate(rotation=45)
    fig2.legend()
# %%
    # Fit the polynomial for p = 0.1n and choose the biggest n
    for p in [ 50, 75, 100, 125, 150, 155, 300]:
        n = p*10
        epsilon = norm.rvs(loc=0,scale=0.11,size=n)
        x = np.array([ (4*np.pi*(i+1))/n for i in range(n) ], dtype=float)
        y = np.sin(x) + epsilon
        coef = polfit(x,y,p-1)
        print("¡Lo logró hasta ",n,"!")
# %%
