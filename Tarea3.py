def compare(X,beta,sigma=0.13):
    '''This function gets the Least Squares Estimator for a regression problem with controlled
    noise using two diffeent methods and adding aproximation error. It is designed to compare
    how different perturbations of input and output con modify the estimation of hat beta with
    each method'''
    # Import needed modules
    import numpy as np
    from scipy.stats import norm as N
    from scipy.linalg import inv
    from QR import LSQR
    
    # Define our variables
    n,d = X.shape
    epsilon = N.rvs(loc=0, scale=sigma, size=n)
    y = X @ beta + epsilon
    DeltaX = N.rvs(loc=0,scale=0.01,size=d*n).reshape((n,d))

    # Solve using QR for Least Squares problem for every case
    hat_beta = LSQR(X,y)
    hat_beta_p = LSQR(X + DeltaX, y)
    hat_beta_c = ( inv( (X + DeltaX).T @ (X + DeltaX) ) @ (X + DeltaX).T ) @ y
    
    # Return the three different estimates for beta
    return hat_beta, hat_beta_p, hat_beta_c

def timer_chol(Matrix,method,times):
    '''This is a custom function created only to evaluate running times of Cholesky decomposition
    algorithms with different matrices'''
    # Import needed modules
    from scipy.linalg import cholesky as schol
    from time import time
    # If choosen methis is scipy, we use Scipy's cholesky
    if method == "scipy":
        # The try-except structure catches errors if there's something wrong
        try:
            start = time()
            # We run the algorithm several times in order to measure the elapsed time
            # (A single time would give a very short time)
            for i in range(times):
                chol = schol(Matrix,overwrite_a=True, check_finite=False)
            end = time()
        except:
            print("Error en la descomposición con Scipy")
    # When the method is not scipy we repeat the same, but using our own method
    else:
        try:
            start = time()
            for i in range(times):
                chol = cholesky(Matrix)
            end = time()
        except:
            print("Error en la descomposición propia")
    return end - start

def up_max(A):
    "This function gets the maximum value of an upper triangular matrix"
    import numpy as np
    # This function creates a tuple with the non zero indices in an upper triangular matrix
    indices = np.triu_indices(A.shape[0])
    # We use the indices to create a list with the above diagonal elements in the matrix
    triang = [A[i][j] for i in indices[0] for j in indices[1]]
    # Finally, we find the maximum and return it
    return np.max(np.abs(triang))

def up_mean(A):
    "This function gets the mean value of an upper triangular matrix"
    import numpy as np
    # This function creates a tuple with the non zero indices in an upper triangular matrix
    indices = np.triu_indices(A.shape[0])
    # We use the indices to create a list with the above diagonal elements in the matrix
    triang = [A[i][j] for i in indices[0] for j in indices[1]]
    # Finally, we find the mean and return it
    return np.mean(triang)

def up_sum(A):
    "This function gets the sum of absolute values of an upper triangular matrix"
    import numpy as np
    # This function creates a tuple with the non zero indices in an upper triangular matrix
    indices = np.triu_indices(A.shape[0])
    # We use the indices to create a list with the above diagonal elements in the matrix
    triang = np.array([A[i][j] for i in indices[0] for j in indices[1]])
    # Finally, we find the sum and return it
    return np.sum( np.abs(triang))

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

    # generate the random matrix
    A = N.rvs(scale=1, loc=0, size=400)
    A = A.reshape((20,20))

    # For a unitary matrix
    Q, R = QR(A)
    
    # Create spaced eigenvalues
    alpha = 7
    lambdas_malas = [(alpha**19) / (alpha**i) for i in range(20)]
    # For good conditioned case we only take evenly spaces between 20 and 1
    lambdas_buenas = np.linspace(20,1,num=20)

    # Random noise N(0,0.02)
    epsilon = N.rvs(scale=0.02, loc=0, size=20)

    # Create B and B_e matrices which are good and ill conditioned
    B_mala = (Q.T @ np.diag(lambdas_malas) ) @ Q
    Be_mala = (Q.T @ (np.diag(lambdas_malas + epsilon))) @ Q

    B_buena = (Q.T @ np.diag(lambdas_buenas) ) @ Q
    Be_buena = (Q.T @ (np.diag(lambdas_buenas + epsilon))) @ Q

    # Find Cholesky decomposition for any case with own algorithm
    Bchol_buena = cholesky(B_buena)
    Bechol_buena = cholesky(Be_buena)
    Bchol_mala = cholesky(B_mala)
    Bechol_mala = cholesky(Be_mala)
    
    # Find Cholesky decomposition with Scipy's algorithm. We use the options to get the
    # best performance
    sBchol_buena = schol(B_buena,overwrite_a=True, check_finite=False)
    sBechol_buena = schol(Be_buena,overwrite_a=True, check_finite=False)
    sBchol_mala = schol(B_mala,overwrite_a=True, check_finite=False)
    sBechol_mala = schol(Be_mala,overwrite_a=True, check_finite=False)

    # Compare B and Be in well conditioned case
    buena_max = up_max(Bechol_buena - Bchol_buena)
    buena_mean = up_mean(Bechol_buena - Bchol_buena)
    buena_sum = up_sum(Bechol_buena - Bchol_buena)

    # Compare B and Be in ill conditioned case with both algorithms
    mala_max = up_max(Bechol_mala - Bchol_mala)
    mala_mean = up_mean(Bechol_mala - Bchol_mala)
    mala_sum = up_sum(Bechol_mala - Bchol_mala)

    smala_max = up_max(sBechol_mala - sBchol_mala)
    smala_mean = up_mean(sBechol_mala - sBchol_mala)
    smala_sum = up_sum(sBechol_mala - sBchol_mala)

    
    # In the next part, we are going to measure the time elapsed for Cholesky in both
    # my own implementation and Scipy's one

    # Variable for number of times every function is ran
    n_times = 1000

    # Get times for the decomposition of every case with own algorithm
    tB_buena = timer_chol(B_buena,"any",n_times)
    tBe_buena = timer_chol(Be_buena,"any",n_times)
    tB_mala = timer_chol(B_mala,"any",n_times)
    tBe_mala = timer_chol(Be_mala,"any",n_times)
    
    # Save all of the times for own algorithm in a vector
    t_hands = np.array( [tB_buena,tBe_buena,tB_mala,tBe_mala] )

    # get the times again, this time for every matrix with scipy's algorithm
    stB_buena = timer_chol(B_buena,"scipy",n_times)
    stBe_buena = timer_chol(Be_buena,"scipy",n_times)
    stB_mala = timer_chol(B_mala,"scipy",n_times)
    stBe_mala = timer_chol(Be_mala,"scipy",n_times)

    # Save all of Scipy's times in a vector
    t_scipy = np.array( [stB_buena,stBe_buena,stB_mala,stBe_mala] )

                            ###### Problema 2 ######
    # Define our variables
    d = 5
    n = 20
    beta = np.array([5,4,3,2,1])
    sigma = 0.13

    # Create well conditioned random matrix
    X = uniform.rvs(size = d*n)
    X = X.reshape((n,d))

    # Use the previously defined function to get the three estimators for beta
    hat_beta, hat_beta_p, hat_beta_c = compare(X,beta,sigma)

    # Create an ill condictioned matrix and again get the three beta estimators for
    # this matrix
    # Generate a smaller matrix
    X_mala = uniform.rvs(size = int(3*n)).reshape((n,3))
    # Create two new columns as linear combinations and add a little random noise
    ncol1 = 0.5*X_mala[:,0] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    ncol2 = 0.5*X_mala[:,1] + 0.5*X_mala[:,2] + N.rvs(loc=0,scale=0.00001,size=20)
    # Attach the new columns to the smaller matrix to creat an ill conditioned one
    X_mala = np.c_[X_mala, ncol1,ncol2]

    hat_beta_mala, hat_beta_p_mala, hat_beta_c_mala = compare(X_mala,beta,sigma)