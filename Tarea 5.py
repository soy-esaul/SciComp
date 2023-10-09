def simulate_unif(n,seed=[57,189,42,26,4]):
    '''This function simulates a sample of n i.i.d. random variables distributed
    as uniform in (0,1)
    
    Arguments:
    - n: Number of variables to simulate
    - seed: A list or array-like of lenght 5 with the initial states for the algorithm
    
    Output:
    - u: A vector with the simulated values (numpy array of shape (1,n))'''
    import numpy as np
    seed = np.asarray(seed, dtype=float)
    uniform_vector = np.zeros((n,1))
    for i in range(n):
        uniform_vector[i] = (107374182*seed[4] + 104420*seed[0]) % (2**(31) - 1)
        seed = np.append(seed[1:],x[i])
    return uniform_vector / (2**(31) - 2)
# Simulation of exponential e = -log(u)

def log_gamma_dens(point,alpha=2,beta=1):
    '''This function evaluates the logarithm of a Gamma(alpha,beta) density at 
    a given point'''
    import numpy as np
    from scipy.stats import gamma
    from scipy.special import gamma as Gamma
    return (alpha*np.log(beta) - np.log(Gamma(alpha)) + (alpha-1)*np.log(point) - beta*point)

def create_line(start,end,point_to_evaluate):
    '''This function finds the function of a straight line passing through two points
    and evaluates it at a given value of x
    
    Arguments:
    - start: A 2-tuple with the x and y coordinates of the first point to find the line
    - end: A 2-tuple with the x and y coordinates of the second point to find the line'''
    return ((end[1]-start[1])*point_to_evaluate + start[1]*end[0] - start[0]*end[1] )/ (end[0] - start[0])

def envelope(points,x):
    '''This function creates an envelope for the logarithm of a Gamma density
    
    Arguments:
    - points: A list or array-like containing the set of points to be used 
    in the envelope'''
    import numpy as np
    x = np.sort(x)
    if x >= 0 and x <= points[0]:
        value = create_line((points[0],log_gamma_dens(points[0])),(points[1],log_gamma_dens(points[1])),x)
    elif x >= points[-1]:
        value = create_line((points[-2],log_gamma_dens(points[-2])),(points[-1],log_gamma_dens(points[-1])),x)
    elif x < 0:
        print("Error: La entrada debe ser no negativa")
    elif x > points[0] and x <= x[1]:
        value = create_line((points[1],log_gamma_dens(points[1])),(points[2],log_gamma_dens(points[2])),x)
    elif x > points[-2] and x < points[-1]:
        value = create_line((points[-3],log_gamma_dens(points[-3])),(points[-2],log_gamma_dens(points[-2])),x)
    else:
        pos = len([i for i in points if i <= x]) - 1
        value_1 = create_line((points[pos],log_gamma_dens(points[pos])),(),x )
        value_2 = create_line((),(),x )
    
    
if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    
    