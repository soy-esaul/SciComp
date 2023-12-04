import numpy as np
import matplotlib.pyplot as plt

# Functions
## For exercise 1
def logdensmvn(x,mu,Sigma=np.array([[1,0],[0,1]],dtype=float)):
    mu = np.asarray(mu,dtype=float)
    detSigma = Sigma[0,0]*Sigma[1,1] - Sigma[0,1]*Sigma[1,0]
    iSigma = np.array([[Sigma[1,1],-Sigma[0,1]],[-Sigma[1,0],Sigma[0,0]]]) / detSigma
    return -log(2*pi)-(1/2)*(log(detSigma) + (x-mu).T @ iSigma @ (x-mu))

def logdensnorm(x,mu,sigma):
    return (-1/2)*(log(2*pi*sigma)+(((x-mu)**2)/(sigma)))

def densmvn(x,y,rho,mux=0,muy=0,sigma1=1,sigma2=2):
    f = 1/(pi*sigma1*sigma2*np.sqrt(1-rho**2))*exp(-(((x-mux)/sigma1)**2 + 
        ((y-muy)/sigma2)**2 - 2*rho*(x-mux)*(y-muy)/(sigma1 * sigma2))
        /(2*(1-rho**2)))
    return f

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
            current_point[0] = proposal.copy()
        else:
            proposal = normal.rvs(loc=mu[1]+rho*((sigma2/sigma1)*(current_point[0]-mu[0])),
                               scale=sigma2*(1-rho**2))
            current_point[1] = proposal.copy()
        output.append(current_point.copy())
    output = np.array(output,dtype=float)
    return output

# def mvn_graph(x,mu,Sigma=np.array([[1,0],[0,1]],dtype=float)):

        
## For exercise 2
def logdenspost(alpha,lamda,t,n=20,c=1,b=1):
    return (n*(log(alpha) + log(lamda)) + (alpha-1)*sum(log(t)) - 
            lamda*sum(t**alpha)-alpha*c+(alpha-1)*log(lamda)
            +alpha*log(b)-b*lamda-log(gamma(alpha)))

def loggammadens(x,alpha,beta):
    return (-log(gamma(alpha)) + alpha*log(beta) + (alpha-1)*log(x) - beta*x)


def MHposter(data,iterations=1000,starting_point=[1,1],n=20,b=1,c=1,sigma=1):
    output = []
    current_point = np.array(starting_point,dtype=float)
    data = np.array(data,dtype=float)
    output.append(current_point)
    for t in range(iterations):
        alpha = current_point[0]
        lamda = current_point[1]
        u = np.random.random()
        if u < 0.25:
            proposal =  gammavar.rvs(a=alpha+n,scale=1/(b+sum(data**alpha)))
            current_point[1] = proposal.copy()
        elif u >= 0.25 and u < 0.5:
            if c -log(b) - sum(log(data)) > 0:
                proposal = gammavar.rvs(a=n+1,scale=1/(-log(b)-sum(log(data))+c))
                Vp = logdenspost(proposal,lamda,data,n)
                V  = logdenspost(alpha,lamda,data,n)
                Wp = loggammadens(proposal,n+1,-log(b)-sum(log(data))+c)
                W  = loggammadens(alpha,n+1,-log(b)-sum(log(data))+c)
                unif = np.random.random()
                if log(unif) < Vp-V+W-Wp:
                    current_point[0] = proposal.copy()
            else:
                pass
        elif u >= 0.5 and u < 0.75:
            alpha_p = expon.rvs(0)
            lamda_p = gammavar.rvs(a=alpha_p,scale=1/b)
            Vp = logdenspost(alpha_p, lamda_p, data, n)
            V  = logdenspost(alpha,   lamda,   data, n)
            Wp = loggammadens(lamda_p, alpha_p, b) + (log(c) - c*alpha_p)
            W  = loggammadens(lamda,   alpha,   b) + (log(c) - c*alpha)
            unif = np.random.random()
            if log(unif) < Vp-V+W-Wp:
                current_point = np.array([alpha_p,lamda_p],dtype=float)
        else:
            epsilon = normal.rvs(loc=0,scale=sigma)
            proposal = alpha + epsilon
            Vp = logdenspost(proposal,lamda,data,n)
            V  = logdenspost(alpha,lamda,data,n)
            unif = np.random.random()
            if log(unif) < Vp-V:
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
    from scipy.stats import weibull_min 
    
    import numpy as np
    from numpy.linalg import det
    from numpy.linalg import inv
    from numpy import sum
    from numpy import log
    from numpy import pi
    from numpy import exp
    from numpy import prod

    np.random.seed(17)

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use("seaborn-v0_8")

    # Exercise 1

    Sigma1 = np.array([[1,0.8],[0.8,1]],dtype=float)
    Sigma2 = np.array([[1,0.95],[0.95,1]],dtype=float)
    mvn_sample1 = mhbivar(iterations=10000,mu=[2, 3],rho=0.8)
    mvn_sample2 = mhbivar(iterations=10000,mu=[3,-2],rho=0.95)

    delta = 0.01
    mvn1_x = np.arange(-1,5,delta)
    mvn1_y = np.arange(0,6,delta)
    X1, Y1 = np.meshgrid(mvn1_x, mvn1_y)
    Z1 = densmvn(X1,Y1,rho=0.8,mux=2,muy=3,sigma1=1,sigma2=1)
    plt.contour(X1,Y1,Z1,levels=20,cmap="viridis")
    plt.plot(mvn_sample1[0,0],mvn_sample1[0,1],'ro',label="Punto inicial")
    plt.plot(mvn_sample1[:,0],mvn_sample1[:,1],alpha=0.75,marker='.', linewidth=0.5, markersize=0.75)
    plt.title(r"Densidad normal multivariada con $\rho=0.8$")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.legend()
    plt.savefig("Tarea8/traj11.png")
    plt.show()

    mvn2_x = np.arange(0,6,delta)
    mvn2_y = np.arange(-5,1,delta)
    X2, Y2 = np.meshgrid(mvn2_x, mvn2_y)
    Z2 = densmvn(X2,Y2,rho=0.95,mux=3,muy=-2,sigma1=1,sigma2=1)
    plt.contour(X2,Y2,Z2,levels=10,cmap="viridis")
    plt.plot(mvn_sample2[0,0],mvn_sample2[0,1],'ro',label="Punto inicial")
    plt.plot(mvn_sample2[:,0],mvn_sample2[:,1],alpha=0.75,marker='.', linewidth=0.5, markersize=0.75)
    plt.title(r"Densidad normal multivariada con $\rho=0.95$")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.legend()
    plt.savefig("Tarea8/traj12.png")
    plt.show()

    axis = np.linspace(-5,5,500)
    label1 = r"Marginal en $x$ para $\rho=0.8$, $\mu=2$"
    label2 = r"Marginal en $y$ para $\rho=0.8$, $\mu=3$"
    label3 = r"Marginal en $x$ para $\rho=0.95$, $\mu=3$"
    label4 = r"Marginal en $y$ para $\rho=0.95$, $\mu=-2$"
    plt.hist(mvn_sample1[:,0],density=True,bins=50,label=label1,alpha=0.8)
    yaxis1 = normal.pdf(axis,loc=2,scale=np.sqrt(1-0.8**2))
    yaxis2 = normal.pdf(axis,loc=3,scale=np.sqrt(1-0.8**2))
    yaxis3 = normal.pdf(axis,loc=3,scale=np.sqrt(1-0.95**2))
    yaxis4 = normal.pdf(axis,loc=-2,scale=np.sqrt(1-0.95**2))
    plt.hist(mvn_sample1[:,1],density=True,bins=50,label=label2,alpha=0.8)
    plt.hist(mvn_sample2[:,0],density=True,bins=50,label=label3,alpha=0.8)
    plt.hist(mvn_sample2[:,1],density=True,bins=50,label=label4,alpha=0.8)
    plt.plot(axis,yaxis1)
    plt.plot(axis,yaxis2)
    plt.plot(axis,yaxis3)
    plt.plot(axis,yaxis4)
    plt.title(r"Histogramas de las marginales")
    plt.legend()
    plt.savefig("Tarea8/hist_dens.png")
    plt.show()

    # Burn-in
    logtrajex1 = []
    sigma1 = np.array([[1,  0.8], [0.8,  1]], dtype=float)
    for i in range(int(np.shape(mvn_sample1)[0])):
        x = np.array([mvn_sample1[i,0],mvn_sample1[i,1]],dtype=float)
        logpoint = logdensmvn(x,mu=[2,3],Sigma=sigma1)
        logtrajex1.append(logpoint)
    logtrajex1 = np.array(logtrajex1,dtype=float)
    plt.plot(logtrajex1)
    plt.xlabel("Iteraciones")
    plt.ylabel("logaritmo de la densidad")
    plt.title(r"Trayectoria para burn-in del ejercicio 1 con $\rho=0.8$")
    plt.savefig("burnin1.png")
    plt.show()

    logtrajex2 = []
    sigma2 = np.array([[1, 0.95], [0.95, 1]], dtype=float)
    for i in range(int(np.shape(mvn_sample2)[0])):
        x = np.array([mvn_sample2[i,0],mvn_sample2[i,1]],dtype=float)
        logpoint = logdensmvn(x,mu=[3,-2],Sigma=sigma2)
        logtrajex2.append(logpoint)
    logtrajex2 = np.array(logtrajex2,dtype=float)
    plt.plot(logtrajex2)
    plt.xlabel("Iteraciones")
    plt.ylabel("logaritmo de la densidad")
    plt.title(r"Trayectoria para burn-in del ejercicio 1 con $\rho=0.95$")
    plt.savefig("burnin2.png")
    plt.show()


    # Exercise 2
    data = weibull_min.rvs(c=1,size=20)
    fail_sims = MHposter(data,50000)
    print(np.mean(fail_sims[:,0]),np.mean(fail_sims[:,1]))
    plt.plot(fail_sims[:,0],fail_sims[:,1],alpha=0.75,marker='.',
             linewidth=0.5, markersize=0.75)
    plt.plot(fail_sims[0,0],fail_sims[0,1],'ro',label="Punto inicial")
    plt.legend()
    plt.title("Simulación de la cadena MCMC para los tiempos de falla")
    plt.savefig("Tarea8/traj2.png")
    plt.show()

    plt.hist(fail_sims[:,0],density=True,bins=50,alpha=0.75)
    plt.title(r"Histograma para $\alpha$")
    plt.savefig("Tarea8/hist21.png")
    plt.show()

    plt.hist(fail_sims[:,1],density=True,bins=50,alpha=0.75,color="teal")
    plt.title(r"Histograma para $\lambda$")
    plt.savefig("Tarea8/hist22.png")
    plt.show()

    # Burn-in
    logtrajex3 = []
    for i in range(int(np.shape(fail_sims)[0])):
        logpoint = logdenspost(alpha=fail_sims[i,0],lamda=fail_sims[i,1],t=data)
        logtrajex3.append(logpoint)
    logtrajex3 = np.array(logtrajex3,dtype=float)
    plt.plot(logtrajex3)
    plt.xlabel("Iteraciones")
    plt.ylabel("logaritmo de la densidad")
    plt.title("Trayectoria para burn-in del ejercicio 2")
    plt.savefig("burnin3.png")
    plt.show()


    # Exercise 3
    
    pump_sims = MHpump(iterations=50000)
    params = ["\lambda_1","\lambda_2","\lambda_3","\lambda_4","\lambda_5","\lambda_6",
              "\lambda_7","\lambda_8","\lambda_9","\lambda_{10}",r"\beta"]
    for i in range(11):
        plt.hist(pump_sims[:,i],density=True,bins=50,alpha=0.75)
        plt.title(f"Histograma de ${params[i]}$")
        plt.savefig(f"Tarea8/hist{i}.png")
        plt.show()