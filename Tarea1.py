import numpy as np
from LUyCholesky import LUP, cholesky, backsubs, forsubs

if __name__ == "__main__":
    ## For replicability
    np.random.seed(57)
    ## Assigned matrix
    A = np.matrix('1,0,0,0,1;-1,1,0,0,1;-1,-1,1,0,1;-1,-1,-1,1,1;-1,-1,-1,-1,1'
        ,dtype=float)
    
    # Matrix with the random vectors
    b = np.random.random((5,5))
    # Run if the system is compatible determinated
    if np.linalg.det(A) != 0:
        x = np.zeros((5,5))
        L, U, P = LUP(A,out="copy")
        for i in range(b.shape[1]):
            y = forsubs(L,b[:,i])
            x[:,i] = backsubs(U,y)

    # Comparison of times
    import time
    # Random matrix
    m = 350
    t_chol = np.zeros((m,))
    t_LU = np.zeros((m,))
    for i in range(m):
        h = np.random.random((i,i))
        h = h @ h.T
        start = time.time()
        L, U, P = LUP(h)
        end = time.time()
        t_LU[i] = end - start
        start = time.time()
        R = cholesky(h,copy=True)
        end = time.time()
        t_chol[i] = end - start

    count = sum( t_chol < t_LU )

    # Plot of time comparison
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("pgf")
    matplotlib.style.use("seaborn-v0_8")
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,})
    
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(t_LU,label=r"LUP")
    ax.plot(t_chol,label=r"Cholesky")
    fig.suptitle("Tiempo de ejecución de algoritmos Cholesky y LUP")
    ax.set_xlabel(r"$n$ (Tamaño de la matriz)")
    ax.set_ylabel(r"Tiempo (segundos)")
    fig.legend()
    plt.savefig('LU-Chol.pgf')