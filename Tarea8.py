# Functions
## For exercise 1
def logdensmvn(x,mu,Sigma=np.array([[1,0],[0,1]])):
    x = np.asarray(x,dtype=float)
    mu = np.asarray(mu,dtype=float)
    iSigma = np.array([[-Sigma[1,1],-Sigma[0,1]],[-Sigma[1,0],Sigma[0,0]]])/det(Sigma)
    return -log(2*pi)-(1/2)*(log(abs(det(Sigma)))+(x-mu).T@iSigma@(x-mu))

def logdensnorm(x,mu,sigma):
    return -(1/2)*log(2*pi)-(1/2)*log(sigma)-(((x-mu)**2)/(2*sigma))

def mhbivar(iterations,mu=[0,0],rho=0.8,starting_point=np.array([0,0])):
    '''Implementation of Metropolis-Hastings algorithm for exercise 1'''
    output = []
    mu = np.array(mu, dtype=float)
    Sigma = np.array([[1,rho],[rho,1]],dtype=float)
    sigma1 = Sigma[0,0]
    sigma2 = Sigma[1,1]
    current_point = starting_point
    for t in range(iterations):
        u = np.random.random()
        if u > 0.5:
            proposal = mvn.rvs(mean=mu[0]+rho*((sigma1/sigma2)*(current_point[1]-mu[1])),cov=sigma1**2*(1-rho**2))
            V = logdensmvn(current_point,mu,Sigma)
            Vp = logdensmvn([proposal,current_point[1]],mu,Sigma)
            Wp = logdensnorm(proposal,mu[0]+rho*(current_point[1]-mu[1]),1-rho**2)
            Wppr = logdensnorm(current_point[1],mu[0]+rho*(proposal-mu[1]),1-rho**2)
            unif = np.random.random()
            if exp(Vp-V+Wppr-Wp) < unif:
                current_point[0] = proposal
                print("yes1")
        else:
            proposal = mvn.rvs(mean=mu[1]+rho*((sigma2/sigma1)*(current_point[0]-mu[0])),cov=sigma2**2*(1-rho**2))
            V  = logdensmvn(current_point,mu,Sigma)
            Vp = logdensmvn([current_point[1],proposal],mu,Sigma)
            Wp = logdensnorm(proposal,mu[1]+rho*(current_point[0]-mu[0]),1-rho**2)
            Wppr = logdensnorm(current_point[0],mu[0]+rho*(proposal-mu[1]),1-rho**2)
            unif = np.random.random()
            if exp(Vp-V+Wppr-Wp) < unif:
                current_point[1] = proposal
                print("yes2")
        output.append(current_point)
    output = np.array(output)
    return output
        
## For exercise 2
def logweidens(t,alpha,l):
    return log(alpha)+log(l)+(alpha-1)*log(t)-(t**alpha*l)

        
## For exercise 3

    

# Examples and homework
if __name__ == "__main__":
    from scipy.stats import multivariate_normal as mvn
    import numpy as np
    from numpy.linalg import det
    from numpy import abs
    from numpy import log
    from numpy import pi
    from numpy import exp