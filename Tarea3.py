def compare(X,beta,sigma=0.13):
    '''This function gets the Least Squares Estimator for a regression problem with controlled
    noise using two diffeent methods and adding aproximation error. It is designed to compare
    how different perturbations of input and output con modify the estimation of hat beta with
    each method'''
    import numpy as np
    from scipy.stats import norm as N
    from scipy.linalg import inv
    from QR import LSQR
    
    epsilon = N.rvs(loc=0, scale=sigma, size=n)
    y = X @ beta + epsilon
    DeltaX = N.rvs(loc=0,scale=0.01,size=d*n).reshape((n,d))

    hat_beta = LSQR(X,y)
    hat_beta_p = LSQR(X + DeltaX, y)
    hat_beta_c = ( inv( (X + DeltaX).T @ (X + DeltaX) ) @ (X + DeltaX).T ) @ y
    
    return hat_beta, hat_beta_p, hat_beta_c
if __name__ == "__main__":
    # %%
                            ###### Problema 1 ######
    # Recquired packages
    import numpy as np
    from QR import QR, LSQR
    from LUyCholesky import LUP, cholesky, backsubs, forsubs
    from scipy.stats import norm as N
    from scipy.stats import uniform
    from scipy.linalg import inv
    from scipy.linalg import cholesky as schol
    from time import time

    A = N.rvs(scale=1, loc=0, size=400)
    A = A.reshape((20,20))

    Q, R = QR(A)

    lambda1 = 5e13
    ratio = 20

    lambdas_malas = [i/ratio for i in np.linspace(lambda1,ratio,num=20) ]
    lambdas_buenas = np.linspace(20,1,num=20)

    epsilon = N.rvs(scale=0.01, loc=0, size=20)

    B = (Q.T @ np.diag(lambdas) ) @ Q
    Be = (Q.T @ (np.diag(lambdas + epsilon))) @ Q

    start = time()
    Bchol = cholesky(B)
    Bechol = cholesky(Be)
    end = time()

    t_hands = end - start

    start = time()
    sBchol = schol(B,overwrite_a=True, check_finite=False)
    sBechol = schol(Be,overwrite_a=True, check_finite=False)
    end = time()

    t_scipy = end - start

    print(t_hands, t_scipy)
    # %%
                            ###### Problema 2 ######
    d = 5
    n = 20

    beta = np.array([5,4,3,2,1])
    sigma = 0.13

    X = uniform.rvs(size = d*n)
    X = X.reshape((n,d))

    hat_beta, hat_beta_p, hat_beta_c = compare(X,beta,sigma)
    
    # %%
    X_mala = uniform.rvs(size = int(3*n)).reshape((n,3))
    ncol1 = 0.5*X_mala[:,0] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    ncol2 = 0.5*X_mala[:,1] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    X_mala = np.c_[X_mala, ncol1,ncol2]

    hat_beta_mala, hat_beta_p_mala, hat_beta_c_mala = compare(X_mala,beta,sigma)