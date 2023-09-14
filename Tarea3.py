# %%
                           ###### Problema 1 ######
# Recquired packages
import numpy as np
from QR import QR, LSQR
from LUyCholesky import LUP, cholesky, backsubs, forsubs
from scipy.stats import norm as N
from scipy.linalg import cholesky as schol
from time import time

A = N.rvs(scale=1, loc=0, size=400)
A = A.reshape((20,20))

Q, R = QR(A)

lambdas = [9e15,8e14,7e13,6e12,5e11,4e10,3e9,2e8,2e7,2e6,2e5,2e4,2e3,2e2,60,5,4,3,2,1]
epsilon = N.rvs(scale=0.01, loc=0, size=20)

B = (Q.T @ np.diag(lambdas) ) @ Q
Be = (Q.T @ (np.diag(lambdas + epsilon))) @ Q

start = time()
Bchol = cholesky(B)
Bechol = cholesky(Be)
end = time()

t_hands = end - start

start = time()
sBchol = schol(B)
sBechol = schol(Be)
end = time()

t_scipy = end - start

print(t_hands, t_scipy)
# %%
                           ###### Problema 2 ######
