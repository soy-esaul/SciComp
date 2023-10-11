# %% Functions for the uniform simulation
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
        seed = np.append(seed[1:],uniform_vector[i])
    return uniform_vector / (2**(31) - 2)

# Functions for the gamma ARS
def log_gamma_dens(point,alpha=2,beta=1):
    '''This function evaluates the logarithm of a Gamma(alpha,beta) density at 
    a given point'''
    import numpy as np
    from scipy.special import gamma as Gamma
    return (alpha*np.log(beta) - np.log(Gamma(alpha)) + (alpha-1)*np.log(point) - beta*point)

def gamma_dens(point,alpha=2,beta=1):
    '''This function evaluates a Gamma density in a given point'''
    import numpy as np
    from scipy.special import gamma as Gamma
    return (((beta**alpha)/Gamma(alpha))*point**(alpha-1)*np.exp(-beta*point))
    

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
    points = np.sort(points)
    if x >= 0 and x <= points[0]:
        value = create_line((points[0],log_gamma_dens(points[0])),(points[1],log_gamma_dens(points[1])),x)
    elif x >= points[-1]:
        value = create_line((points[-2],log_gamma_dens(points[-2])),(points[-1],log_gamma_dens(points[-1])),x)
    elif x < 0:
        print("Error: La entrada debe ser no negativa")
    elif x > points[0] and x <= points[1]:
        value = create_line((points[1],log_gamma_dens(points[1])),(points[2],log_gamma_dens(points[2])),x)
    elif x > points[-2] and x <= points[-1]:
        value = create_line((points[-3],log_gamma_dens(points[-3])),(points[-2],log_gamma_dens(points[-2])),x)
    else:
        pos = len([i for i in points if i <= x]) - 1
        value_1 = create_line((points[pos-1],log_gamma_dens(points[pos-1])),(points[pos],log_gamma_dens(points[pos])),x)
        value_2 = create_line((points[pos+1],log_gamma_dens(points[pos+1])),(points[pos+2],log_gamma_dens(points[pos+2])),x)
        value = np.min([value_1,value_2])
    return value

def exp_envelope(x_values,grid):
    '''This function computes an envelope for a Gamma density through a logarithmic
    envelope'''
    import numpy as np
    if type(x_values) in {int,float}:
        y_values = np.exp(envelope(grid,x_values))
    else:
        y_values = np.zeros(len(x_values))
        for i in range(len(x_values)):
            y_values[i] = envelope(grid,x_values[i])
        y_values = np.exp(y_values)
    return y_values

def sp_envelope_cdf(grid,lim):
    '''This function computes the distribution function of an envelope with 
    scipy.integrate'''
    from scipy.integrate import quad
    import numpy as np
    return quad(exp_envelope,0,lim,args=grid)

def exp_integral(liminf,limsup,a,b):
    '''This function evaluates the integral for exp(ax + b) where a and b are the slope
    and intercept of a rect being part of an envelope'''
    import numpy as np
    coeff = (limsup[1] - liminf[1])/(limsup[0] - liminf[0])
    return (np.exp((-limsup[1]*liminf[0] + liminf[1]*limsup[0]) / (limsup[0] - liminf[0]) ))*(np.exp(coeff*b)-np.exp(coeff*a)) / coeff

def envelope_cdf(grid,x):
    '''This function evaluates the cumulative distribution function for an envelope'''
    import numpy as np
    points = np.sort(grid)
    if x <= 0:
        integral = 0
    elif x > 0 and x <= points[0]:
        integral = exp_integral((points[0],log_gamma_dens(points[0])),(points[1],log_gamma_dens(points[1])),0,x)
    elif x > points[0] and x <= points[1]:
        integral = envelope_cdf(grid,points[0]) + exp_integral((points[1],log_gamma_dens(points[1])),(points[2],log_gamma_dens(points[2])),points[0],x)
    elif x > points[-2] and x <= points[-1]:
        integral = envelope_cdf(grid,points[-2]) + exp_integral((points[-3],log_gamma_dens(points[-3])),(points[-2],log_gamma_dens(points[-2])),points[-2],x)
    elif x > points[-1]:
        integral = envelope_cdf(grid,points[-1]) + exp_integral((points[-2],log_gamma_dens(points[-2])),(points[-1],log_gamma_dens(points[-1])),points[-1],x)
    else:
        pos = len([i for i in points if i < x]) - 1
        logx_i1 = log_gamma_dens(points[pos+1])
        logx_i2 = log_gamma_dens(points[pos+2])
        logx_n1 = log_gamma_dens(points[pos-1])
        logx_i  = log_gamma_dens(points[pos])
        denominator = (logx_i*points[pos-1] - logx_n1*points[pos])/(points[pos]-points[pos-1]) + (- logx_i2*points[pos+1] + logx_i1*points[pos+2]) / (points[pos+2] - points[pos+1])
        numerator = (logx_i - logx_n1)/(points[pos] - points[pos-1]) - (logx_i2 - logx_i1)/(points[pos+2] - points[pos+1])
        midpoint = denominator / numerator
        if x <= midpoint:
            integral = envelope_cdf(grid,points[pos]) + exp_integral((points[pos-1],logx_n1),(points[pos],logx_i),points[pos],x)
        else:
            integral = envelope_cdf(grid,midpoint) + exp_integral((points[pos+1],logx_i1),(points[pos+2],logx_i2),midpoint,x)
    return integral

def generalized_inverse(distr,values,x):
    '''This function finds the generalized inverse of a distribution function for a given percentile'''
    i = 0
    while distr[i] < x:
        i += 1
    return values[i]

def ars_gamma(n_simul,grid=[0.5,1,3,6]):
    '''Thsi function generates a quantity (n_simul) of gamma random variables with parameters
    alpha and beta specified by the user using the Adaptive Rejection algorithm as seen in 
    Robert and Casella "Monte Carlo Statistical Methods" (2004).
    
    The Algorithm requieres to be initialized with a grid of numbers. By default, it uses the
    list [0.5,1,3,6] as these numbers are well spaced for a Gamma(2,1) density'''
    import numpy as np
    simulations = []
    x = np.linspace(0,8,1000)
    distr = []
    for i in x:
        distr.append(envelope_cdf(grid,i))
    distr = distr / distr[-1]
    while len(simulations) < n_simul:
        u_1 = np.random.uniform(size=1)
        dummy_var = generalized_inverse(distr,x,u_1)
        u_2 = np.random.uniform(size=1)
        if np.exp(envelope(grid,dummy_var))*u_2 <= gamma_dens(dummy_var):
            simulations.append(dummy_var)
            if len(simulations) < 10:
                grid.append(dummy_var)
                distr = []
                for i in x:
                    distr.append(envelope_cdf(grid,i))
                distr = distr / distr[-1]
    return simulations

# %% 
# Examples and homework
if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.stats import kstest
    np.random.seed(57)
    
    # Uniform simulation
    unif_sims = simulate_unif(10000)
    unif_mean = np.mean(unif_sims)
    unif_var = np.var(unif_sims)
    unif_max_min = [ np.min(unif_sims), np.max(unif_sims) ]
    # Create histogram
    plt.hist(unif_sims,density=True)
    plt.title('Histograma para 10,00 simulaciones U(0,1)')
    plt.ylabel("Frecuencia relativa")
    plt.xlabel("Valor")
    plt.show()
    # Hypothesis test
    unif_sims = unif_sims.reshape((10000,))
    test = kstest(unif_sims,"uniform")

    # Gamma simulation with envelope
    grid = [0.5,1,3,6]
    x = np.linspace(0,8,1000)
    distr = []
    for i in x:
        distr.append(envelope_cdf(grid,i))
    distr = distr / distr[-1]
    # Simulations for the envelope
    env_simulations=[]
    for i in range(10000):
        u = np.random.uniform(size=1)
        env_simulations.append(generalized_inverse(distr,x,u))
    # Simulations for gamma without updating the envelope
    gamma_sims = []
    while len(gamma_sims) <10000:
        u = np.random.uniform(size=1)
        dummy = generalized_inverse(distr,x,u)
        u2 = np.random.uniform(size=1)
        if np.exp(envelope(grid,dummy))*u2 <= gamma_dens(dummy):
            gamma_sims.append(dummy)
    # Using the function to update the envelope
    reg_gamma_sims = ars_gamma(10000)
    gamma_mean = np.mean(reg_gamma_sims)
    gamma_var = np.var(reg_gamma_sims)
    dens = []
    for i in x:
        dens.append(gamma_dens(i))
    envelope_dens = exp_envelope(x,grid)
    plt.plot(x,dens,label="Densidad gamma(0,1)")
    plt.plot(x,envelope_dens,label="Envolvente inicial")
    plt.hist(reg_gamma_sims,density=True,bins=30,label="Histograma")
    plt.xlabel("Valor")
    plt.ylabel("Densidad")
    plt.legend()
    plt.title("Simulación de una muestra Gamma(2,1) mediante ARS")
    plt.show()
    # Hypothesis testing
    from scipy.stats import gamma as g
    sp_sample = g.rvs(2,size=10000)
    gam_sample = array(reg_gamma_sims).reshape((10000,))
    gamma_test = kstest(gam,sp_sample)
                                                
    
    