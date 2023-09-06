# %%
import numpy as np
from LUyCholesky import LUP, cholesky, backsubs, forsubs

if __name__ == "__main__":
    ## For replicability
    np.random.seed(57)
    ## Assigned matrix
    A = np.matrix('1,0,0,0,1;-1,1,0,0,1;-1,-1,1,0,1;-1,-1,-1,1,1;-1,-1,-1,-1,1'
        ,dtype=float)
    ## Random uniform matrix
    R = np.random.random((5,5))
    
    # Matrix with the random vectors
    b = np.random.random((5,5))
    # Run if the system is compatible determinated
    x = np.zeros((5,5))
    x_R = np.zeros((5,5))
    try:
        L, U, P = LUP(A,out="copy")
        L_R, U_R, P_R = LUP(R, out="copy")
    except:
        print("Program didn't finish! Exception ocurred!")
    b_R = b.copy()
    b_R = P_R @ b_R
    for i in range(b.shape[1]):
        y = forsubs(L,b[:,i])
        y_R = forsubs(L_R,b_R[:,i])
        x[:,i] = backsubs(U,y)
        x_R[:,i] = backsubs(U_R,y_R)
# %%
    # Comparison of times
    import time
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

    # Count times LU took longer than Cholesky
    count = sum( t_chol < t_LU )

    # Plot of time comparison
    import matplotlib
    import matplotlib.pyplot as plt
    # For LaTeX output
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