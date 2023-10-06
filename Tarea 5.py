def mudunif(n,seed=[57,189,42,26,4]):
    '''This function simulates a sample of n i.i.d. random variables distributed
    as uniform in (0,1)
    
    Arguments:
    - n: Number of variables to simulate
    - seed: A list or array-like of lenght 5 with the initial states for the algorithm
    
    Output:
    - u: A vector with the simulated values (numpy array of shape (1,n))'''
    import numpy as np
    seed = np.asarray(seed, dtype=float)
    x = np.zeros((n,1))
    for i in range(n):
        x[i] = (107374182*seed[4] + 104420*seed[0]) % (2**(31) - 1)
        seed = np.append(seed[1:],x[i])
    return x / (2**(31) - 2)
# Simulation of exponential e = -log(u)
def ARS()
if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    
    