def compare(X,beta,sigma=0.13):
    '''This function gets the Least Squares Estimator for a regression problem with controlled
    noise using two diffeent methods and adding aproximation error. It is designed to compare
    how different perturbations of input and output con modify the estimation of hat beta with
    each method'''
    import numpy as np
    from scipy.stats import norm as N
    from scipy.linalg import inv
    from QR import LSQR
    
    n,d = X.shape

    epsilon = N.rvs(loc=0, scale=sigma, size=n)
    y = X @ beta + epsilon
    DeltaX = N.rvs(loc=0,scale=0.01,size=d*n).reshape((n,d))

    hat_beta = LSQR(X,y)
    hat_beta_p = LSQR(X + DeltaX, y)
    hat_beta_c = ( inv( (X + DeltaX).T @ (X + DeltaX) ) @ (X + DeltaX).T ) @ y
    
    return hat_beta, hat_beta_p, hat_beta_c
def timer_chol(Matrix,method,times):
    '''This is a custom function created only to evaluate running times of Cholesky decomposition
    algorithms with different matrices'''
    from scipy.linalg import cholesky as schol
    from time import time
    if method == "scipy":
        try:
            start = time()
            for i in range(times):
                chol = schol(Matrix,overwrite_a=True, check_finite=False)
            end = time()
        except:
            print("Error en la descomposición con Scipy")
    else:
        try:
            start = time()
            for i in range(times):
                chol = cholesky(Matrix)
            end = time()
        except:
            print("Error en la descomposición propia")
    return end - start
if __name__ == "__main__":
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

    alpha = 7

    lambdas_malas = [(alpha**19) / (alpha**i) for i in range(20)]
    lambdas_buenas = np.linspace(20,1,num=20)

    epsilon = N.rvs(scale=0.02, loc=0, size=20)

    B_mala = (Q.T @ np.diag(lambdas_malas) ) @ Q
    Be_mala = (Q.T @ (np.diag(lambdas_malas + epsilon))) @ Q

    B_buena = (Q.T @ np.diag(lambdas_buenas) ) @ Q
    Be_buena = (Q.T @ (np.diag(lambdas_buenas + epsilon))) @ Q

    
    Bchol_buena = cholesky(B_buena)
    Bechol_buena = cholesky(Be_buena)
    Bchol_mala = cholesky(B_mala)
    Bechol_mala = cholesky(Be_mala)
    

    tB_buena = timer_chol(B_buena,"any",500)
    tBe_buena = timer_chol(Be_buena,"any",500)
    tB_mala = timer_chol(B_mala,"any",500)
    tBe_mala = timer_chol(Be_mala,"any",500)


   
    sBchol_buena = schol(B_buena,overwrite_a=True, check_finite=False)
    sBechol_buena = schol(Be_buena,overwrite_a=True, check_finite=False)
    sBchol_mala = schol(B_mala,overwrite_a=True, check_finite=False)
    sBechol_mala = schol(Be_mala,overwrite_a=True, check_finite=False)
    
    stB_buena = timer_chol(B_buena,"scipy",500)
    stBe_buena = timer_chol(Be_buena,"scipy",500)
    stB_mala = timer_chol(B_mala,"scipy",500)
    stBe_mala = timer_chol(Be_mala,"scipy",500)

                            ###### Problema 2 ######
    d = 5
    n = 20
    beta = np.array([5,4,3,2,1])
    sigma = 0.13

    X = uniform.rvs(size = d*n)
    X = X.reshape((n,d))

    hat_beta, hat_beta_p, hat_beta_c = compare(X,beta,sigma)

    X_mala = uniform.rvs(size = int(3*n)).reshape((n,3))
    ncol1 = 0.5*X_mala[:,0] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    ncol2 = 0.5*X_mala[:,1] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    X_mala = np.c_[X_mala, ncol1,ncol2]

    hat_beta_mala, hat_beta_p_mala, hat_beta_c_mala = compare(X_mala,beta,sigma)