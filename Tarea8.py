import numpy as np

# Functions
## For exercise 1
def logdensmvn(x,mu,Sigma=np.array([[1,0],[0,1]],dtype=float)):
    mu = np.asarray(mu,dtype=float)
    detSigma = Sigma[0,0]*Sigma[1,1] - Sigma[0,1]*Sigma[1,0]
    iSigma = np.array([[Sigma[1,1],-Sigma[0,1]],[-Sigma[1,0],Sigma[0,0]]]) / detSigma
    return -log(2*pi)-(1/2)*(log(detSigma) + (x-mu).T @ iSigma @ (x-mu))

def logdensnorm(x,mu,sigma):
    return (-1/2)*(log(2*pi*sigma)+(((x-mu)**2)/(sigma)))

def mhbivar(iterations,mu=[0,0],rho=0.8,starting_point=np.array([0,0],dtype=float)):
    '''Implementation of Metropolis-Hastings algorithm for exercise 1'''
    output = []
    mu = np.array(mu, dtype=float)
    Sigma = np.array([[1,rho],[rho,1]],dtype=float)
    sigma1 = Sigma[0,0]
    sigma2 = Sigma[1,1]
    current_point = np.array(starting_point,dtype=float)
    for t in range(iterations):
        u = np.random.random()
        if u > 0.5:
            proposal = normal.rvs(loc=mu[0]+rho*((sigma1/sigma2)*(current_point[1]-mu[1])),
                                   scale=sigma1*(1-rho**2))
            # V = logdensmvn(current_point,mu,Sigma)
            # Vp = logdensmvn(np.array([proposal,current_point[1]],dtype=float),mu,Sigma)
            # Wp = logdensnorm(proposal,mu[0]+rho*(current_point[1]-mu[1]),1-rho**2)
            # Wppr = logdensnorm(current_point[0],mu[0]+rho*(proposal-mu[1]),1-rho**2)
            # unif = np.random.random()
            # if exp(Vp-V+Wppr-Wp) < unif:
            #    current_point[0] = proposal
            current_point[0] = proposal.copy()
        else:
            proposal = normal.rvs(loc=mu[1]+rho*((sigma2/sigma1)*(current_point[0]-mu[0])),
                               scale=sigma2*(1-rho**2))
            # V  = logdensmvn(current_point,mu,Sigma)
            # Vp = logdensmvn(np.array([current_point[0],proposal],dtype=float),mu,Sigma)
            # Wp = logdensnorm(proposal,mu[1]+rho*(current_point[0]-mu[0]),1-rho**2)
            # Wppr = logdensnorm(current_point[1],mu[1]+rho*(proposal-mu[0]),1-rho**2)
            # unif = np.random.random()
            #if exp(Vp-V+Wppr-Wp) < unif:
            #    current_point[1] = proposal
            current_point[1] = proposal.copy()
        output.append(current_point.copy())
    output = np.array(output,dtype=float)
    return output
        
## For exercise 2
def logdenspost(alpha,lamda,t,n=20,c=1,b=1):
    return (n*(log(alpha) + log(lamda)) + (alpha-1)*log(sum(t)) - 
            lamda*sum(t**alpha) +n*(log(c)-alpha*c+(alpha-1)*log(lamda)
            +alpha*log(b)-b*lamda-log(gamma(alpha))))

def loggammadens(x,alpha,beta):
    return (-log(gamma(alpha)) + alpha*log(beta) + (alpha-1)*log(x) - beta*x)


def MHposter(data,iterations=1000,starting_point=[1,1],n=20,b=1,c=1,sigma=1):
    output = []
    current_point = np.array(starting_point,dtype=float)
    data = np.array(data,dtype=float)
    alpha = current_point[0]
    lamda = current_point[1]
    for t in range(iterations):
        alpha = current_point[0]
        lamda = current_point[1]
        u = np.random.random()
        if u < 0.25:
            proposal =  gammavar.rvs(a=alpha+n,scale=1/(b+sum(data**alpha)))
            V  = logdenspost(alpha,proposal,data,n)
            Vp = logdenspost(alpha,lamda,data,n)
            W  = loggammadens(lamda,alpha+n,b+sum(data**alpha))
            Wp = loggammadens(proposal,alpha+n,b+sum(data**proposal))
            unif = np.random.random()
            if exp(V-Vp+Wp-W) > unif:
                current_point[1] = proposal.copy()
        elif u >= 0.25 and u < 0.5:
            r = np.prod(data)
            if c -log(b) - log(r) > 0:
                proposal = gammavar.rvs(n+1,-log(b)-log(r)+c)
                V  = logdenspost(proposal,lamda,data,n)
                Vp = logdenspost(alpha,lamda,data,n)
                W  = loggammadens(proposal,n+1,-log(b)-log(r)+c)
                Wp = loggammadens(lamda,n+1,-log(b)-log(r)+c)
                unif = np.random.random()
                if exp(V-Vp+Wp-W) > unif:
                    current_point[0] = proposal.copy()
            else:
                pass
        elif u >= 0.5 and u < 0.75:
            alpha_p = expon.rvs(c)
            lamda_p = gammavar.rvs(alpha_p,b)
            V  = logdenspost(alpha_p,lamda_p,data,n)
            Vp = logdenspost(alpha,lamda,data,n)
            W  = loggammadens(lamda_p,alpha_p,b)*(log(c) - c*alpha_p)
            Wp = loggammadens(lamda,alpha,b)*(log(c) - c*alpha)
            unif = np.random.random()
            if exp(V-Vp+Wp-W) > unif:
                current_point = np.array([alpha_p,lamda_p],dtype=float) 
        else:
            epsilon = normal.rvs(loc=0,scale=sigma)
            proposal = alpha + epsilon
            V  = logdenspost(proposal,lamda,data,n)
            Vp = logdenspost(alpha,lamda,data,n)
            unif = np.random.random()
            if V-Vp > log(unif):
                current_point[0] = proposal.copy()
        output.append(current_point.copy())
    output = np.array(output,dtype=float)
    return output

def simweibull(alpha,lamda,size):
    x = expon.rvs(lamda,size=size)
    return (x**(1/alpha))
        
## For exercise 3

def MHpump(starting_point=np.array([1,1,1,1,1,1,1,1,1,1,1],dtype=float),
           iterations=100,alpha=1.8,gamita=0.01,delta=1):
    output = []
    t = np.array([94.32,15.72,62.88,125.76,5.24,31.44,1.05,1.05,2.1,10.48],dtype=float)
    p = np.array([5,1,5,14,3,17,1,1,4,22],dtype=float)
    current_point = np.array(starting_point,dtype=float)
    n = 10
    output.append(starting_point)
    for m in range(iterations):
        i = np.random.choice(11,1)[0]
        if i == 10:
            proposal = gammavar.rvs(a=n*alpha+gamita,scale=1/(delta+sum(current_point[:-1])))
            current_point[10] = proposal.copy()
        else:
            beta = current_point[10]
            proposal = gammavar.rvs(a=p[i]+alpha,scale=1/(beta+t[i]))
            current_point[i] = proposal.copy()
        output.append(current_point.copy())
    output = np.array(output,dtype=float)
    return output

    

# Examples and homework
if __name__ == "__main__":
    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import norm as normal
    from scipy.special import gamma
    from scipy.stats import gamma as gammavar
    from scipy.stats import expon
    import numpy as np
    from numpy.linalg import det
    from numpy import sum
    from numpy import log
    from numpy import pi
    from numpy import exp
    from numpy import prod

    np.random.seed(57)

    # Exercise 1


    # Exercise 2
    data = simweibull(1,1,20)
    fail_sims = MHposter(data,10000)
    plt.plot(fail_sims[:,0],fail_sims[:,1])


    # Exercise 3
    pump_starting_point = np.array([94.32,15.72,62.88,125.76,5.24,31.44,1.05,1.05,2.1,10.48,4e-2],
                dtype=float)