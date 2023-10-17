# Definition of functions
def posterior_prop(p,n,r):
    '''This function evaluates the given posterior (without normalizing) for any p 
    and n and r the number of trials and successes respectively'''
    import numpy as np
    if p < 0 or p > 1:
        print("Error: p debe estar entre 0 y 1")
    elif p < 1/2:
        return (p**r)*((1-p)**(n-r))*np.cos(np.pi*p)
    else:
        return 0

def rho_beta(x,y):
    '''This function evaluates the probability of acceptance for the proposal in the 
    specific MCMC algorithm of problem 2'''
    import numpy as np
    rho = np.min([1,np.cos(np.pi*y)/np.cos(np.pi*x)])
    return rho

def MH_beta(iterations,trials,successes,history=False):
    '''This function implements the Metropolis-Hastings algorithm to simulate from 
    the posterior of the parameter in Bayesian inference'''
    import numpy as np
    # Starting point uniform on (0,1/2)
    current_point = np.random.random()*0.5
    output = []
    output.append(current_point)
    if history == False:
        for i in range(iterations):
            proposal = np.random.beta(successes+1,trials-successes+1)
            rho = rho_beta(current_point,proposal)
            u = np.random.random()
            if u <= rho:
                current_point = proposal
                output.append(current_point)
        return output
    else:
        history = []
        for i in range(iterations):
            proposal = np.random.beta(successes+1,trials-successes+1)
            history.append(current_point)
            rho = rho_beta(current_point,proposal)
            u = np.random.random()
            if u <= rho:
                current_point = proposal
                output.append(current_point)
        return output, history

def rho_unif(x,y,n,r):
    '''This function evaluates the probability of acceptance for the proposal in the 
    specific MCMC algorithm of problem 4'''
    import numpy as np
    quotient = ( posterior_prop(y,n,r))/( posterior_prop(x,n,r))
    rho = np.min([1,quotient])
    return rho

def MH_unif(iterations,trials,successes,history=False):
    '''This function implements the Metropolis-Hastings algorithm to simulate from 
    the posterior of the parameter in Bayesian inference'''
    import numpy as np
    # Starting point uniform on (0,1/2)
    current_point = np.random.random()*0.5
    output = []
    output.append(current_point)
    if history == False:
        for i in range(iterations):
            proposal = np.random.random()
            rho = rho_unif(current_point,proposal,trials,successes)
            u = np.random.random()
            if u <= rho:
                current_point = proposal
                output.append(current_point)
        return output
    else:
        history = []
        for i in range(iterations):
            proposal = np.random.random()
            history.append(current_point)
            rho = rho_unif(current_point,proposal,trials,successes)
            u = np.random.random()
            if u <= rho:
                current_point = proposal
                output.append(current_point)
        return output, history

# Examples and homework
if  __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(57)
    
    # Samples of Bernoulli
    sample5 = np.random.binomial(5,1/3,1)
    sample40 = np.random.binomial(40,1/3,1)
    r5 = np.sum(sample5)
    r40 = np.sum(sample40)

    # Implementation with data
    MH_beta_sims5, history_beta_5 = MH_beta(1000,5,r5,history=True)
    MH_beta_sims40, history_beta_40 = MH_beta(1000,40,r40,history=True)
    beta_sizes = [len(MH_beta_sims5), len(MH_beta_sims40)]
    # Histograms

    # Implementation with the new proposal
    MH_unif_sims5, history_unif_5 = MH_unif(1000,5,r5,history=True)
    MH_unif_sims40, history_unif_40 = MH_unif(1000,40,r40,history=True)
    unif_sizes = [len(MH_unif_sims5),len(MH_unif_sims40)]

    # Simple example of MH_beta()
    simple_sample = MH_beta(100,50,28)
    plt.hist(simple_sample,density=True)


