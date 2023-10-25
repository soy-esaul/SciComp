###############################
######## Exercise 1 ###########
###############################

def poster_graf(alpha,beta,data):
    '''This function evaluates the posterior density with a given alpha, beta and some
    data points'''
    n = np.max(np.shape(data))
    r1 = np.prod(data)
    r2 = np.sum(data)
    return (beta**(n*alpha) / Gamma(alpha)**n)*(r1**(alpha-1))*np.exp(-beta*(r2+1))

def poster(alpha,beta,data):
    '''This function evaluates the posterior density with a given alpha, beta and some
    data points'''
    n = int(np.max(np.shape(data)))
    r1 = np.prod(data)
    r2 = np.sum(data)
    if (1 <= alpha <= 4) and (beta > 0):
        post = (beta**(n*alpha) / Gamma(alpha)**n) * (r1**(alpha-1)) * np.exp(-beta*(r2+1))
        return post
    else:
        return 0

def rho_1(x,y,data):
    '''This function evaluates the probability of acceptance for the proposal in the 
    specific MCMC algorithm of problem 1'''
    rho = min(1,(poster(y[0],y[1],data)) / (poster(x[0],x[1],data)))
    return rho

def MHgamma(data,iterations,sigma1=1,sigma2=1):
    '''Implementation of Metropolis-Hastings algorithm for the first excercise'''
    # current_point = np.array( [ np.random.uniform(low=1,high=2.5),np.random.uniform(low=0,high=100)])
    current_point = [3,100]
    alphas = [current_point[0]]
    betas = [current_point[1]]
    for i in range(iterations):
        proposal = np.array([current_point[0] + np.random.normal(loc=0,scale=sigma1), current_point[1] + np.random.normal(loc=0,scale=sigma2)])
        rho = rho_1(current_point,proposal,data)
        u = np.random.uniform()
        if u <= rho:
            current_point = proposal
        alphas.append(current_point[0])
        betas.append(current_point[1])
    return alphas, betas

def MHgamma_arc(data,iterations,sigma1=1,sigma2=1):
    '''Implementation of alternative Metropolis-Hastings algorithm for the first excercise'''
    current_point = [3,100]
    alphas = [current_point[0]]
    betas = [current_point[1]]
    for i in range(iterations):
        proposal = np.array([current_point[0] + semicircular.rvs(scale=sigma1), current_point[1] + semicircular.rvs(scale=sigma2)])
        rho = rho_1(current_point,proposal,data)
        u = np.random.uniform()
        if u <= rho:
            current_point = proposal
        alphas.append(current_point[0])
        betas.append(current_point[1])
    return alphas, betas

    ###############################
    ######## Exercise 2 ###########
    ###############################

    # Original 

def MHuga(iterations,alpha):
    '''Implementation of Metropolis-Hastings algorithm for the second excercise'''
    n = int(alpha)
    current_point = 900
    output = [current_point]
    for i in range(iterations):
        proposal = np.random.gamma(n,scale=1)
        rho = (gamma.pdf(proposal,alpha,scale=1)*gamma.pdf(current_point,n,scale=1) )/ (gamma.pdf(proposal,n,scale=1)*gamma.pdf(current_point,alpha,scale=1))
        u = np.random.uniform()
        if u <= min(1,rho):
            current_point = proposal
        output.append(current_point)
    return output

    # Alternative

def MHuga_alt(iterations,alpha,step=1,starting_point=10):
    '''Implementation of alternative Metropolis-Hastings algorithm for the second excercise'''
    n = int(alpha)
    current_point = starting_point
    output = [current_point]
    for i in range(iterations):
        proposal = 12*np.random.uniform()
        if gamma.pdf(proposal,alpha,scale=1) == 0:
            rho = 0
        else:
            rho = (gamma.pdf(proposal,alpha,scale=1))/(gamma.pdf(current_point,alpha,scale=1))
        u = np.random.uniform(0,1)
        if u <= min(1,rho):
            current_point = proposal
        output.append(current_point)
    return output




    ###############################
    ######## Exercise 3 ###########
    ###############################

    # Original

def rwmh(iterations,sigma,starting_point=[1000,1]):
    '''Implementation of Metropolis-Hastings Random Walk for the third excercise'''
    current_point = np.asarray(starting_point,dtype=float)
    norm1 = [current_point[0]]
    norm2 = [current_point[1]]
    mu = np.array([3,5],dtype=float)
    Sigma = np.array([[1, 0.9],[0.9, 1]], dtype=float)
    mu2 = mu
    Sigma2 = sigma*np.identity(2)
    for i in range(iterations):
        proposal = current_point + mvn.rvs(mean=[0,0],cov=Sigma2,size=1)
        rho = ((mvn.pdf([proposal[0],proposal[1]],mu,Sigma)*mvn.pdf([current_point[0],current_point[1]],proposal,Sigma2)) / 
               (mvn.pdf([current_point[0],current_point[1]],mu,Sigma)*mvn.pdf([proposal[0],proposal[1]],current_point,Sigma2)))
        u = np.random.uniform()
        if u <= rho:
            current_point = proposal
        norm1.append(current_point[0])
        norm2.append(current_point[1])
        mu2 = mu2 + current_point
    return norm1, norm2

    # Alternative

def rwmh_alt(iterations,sigma,starting_point=[10,10]):
    '''Implementation of an alternative Metropolis-Hastings Random Walk for the third excercise'''
    current_point = np.asarray(starting_point,dtype=float)
    norm1 = [current_point[0]]
    norm2 = [current_point[1]]
    mu = np.array([3,5],dtype=float)
    Sigma = np.array([[1, 0.9],[0.9, 1]], dtype=float)
    for i in range(iterations):
        proposal = current_point + mult.rvs(loc=[0,0],df=15)
        rho = ((mvn.pdf([proposal[0],proposal[1]],mu,Sigma))/(mvn.pdf([current_point[0],current_point[1]],mu,Sigma)))
        u = np.random.uniform()
        if u <= rho:
            current_point = proposal
        norm1.append(current_point[0])
        norm2.append(current_point[1])
    return norm1, norm2



################################
#### Solutions to exercises ####
################################

if __name__ == "__main__":
    import numpy as np
    from scipy.special import gamma as Gamma
    from scipy.stats import semicircular
    from scipy.stats import gamma
    from scipy.stats import norm
    from scipy.stats import multivariate_t as mult
    from scipy.stats import multivariate_normal as mvn
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.style.use("seaborn-v0_8")
    plt.rcParams["figure.figsize"] = (7,4)
    np.random.seed(57)

    ###############################
    ######## Exercise 1 ###########
    ###############################
    
    data4 = np.random.gamma(shape=3,scale=(1/100),size=4)
    data30 = np.random.gamma(shape=3,scale=(1/100),size=30)
    delta = 0.01

    alpha4_sims, beta4_sims = MHgamma(data4,30000,sigma1=0.05,sigma2=0.5)
    plt.plot(alpha4_sims,beta4_sims,alpha=0.75,marker='.', linewidth=0.5, markersize=0.75)
    alpha_ax4 = np.arange(1,4,delta)
    beta_ax4 = np.arange(0,15,delta)
    X4, Y4 = np.meshgrid(alpha_ax4, beta_ax4)
    Z4 = poster_graf(X4,Y4,data4)
    plt.contour(X4,Y4,Z4,levels=50,cmap="viridis")
    plt.plot(alpha4_sims[0],beta4_sims[0],'ro')
    plt.title("Trayectoria de la cadena para la posterior con $n=4$")
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\alpha$")
    plt.show()

    # Contour plot
    plt.contour(X4,Y4,Z4,levels=50,cmap="viridis")
    plt.title("Contorno de la posterior con $n=4$")
    plt.show()

    alpha30_sims, beta30_sims = MHgamma(data30,30000,sigma1=0.05,sigma2=0.5)
    plt.plot(alpha30_sims,beta30_sims,alpha=0.75,marker='.', linewidth=0.5, markersize=0.75)
    alpha_ax30 = np.arange(1,4,delta)
    beta_ax30 = np.arange(0,30,delta)
    X30, Y30 = np.meshgrid(alpha_ax30, beta_ax30)
    Z30 = poster_graf(X30,Y30,data30)
    plt.contour(X30,Y30,Z30,levels=50,cmap="viridis")
    plt.plot(alpha30_sims[0],beta30_sims[0],'ro')
    plt.title("Trayectoria de la cadena para la posterior con $n=30$")
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\alpha$")
    plt.show()

    # Contour plot
    plt.contour(X30,Y30,Z30,levels=50,cmap="viridis")
    plt.title("Contorno de la posterior con $n=30$")
    plt.show()

    # Logarithm of density plots
    densgamma4 = []
    for i in range(len(alpha4_sims)):
        densgamma4.append(poster_graf(alpha4_sims[i],beta4_sims[i],data4))
    plt.plot(np.log(densgamma4),linewidth=0.75)
    plt.title("Logaritmo de $f(X_t)$ para $n=4$")
    plt.show()
    
    densgamma30 = []
    for i in range(len(alpha30_sims)):
        densgamma30.append(poster_graf(alpha30_sims[i],beta30_sims[i],data30))
    plt.plot(np.log(densgamma30),linewidth=0.75)
    plt.title("Logaritmo de $f(X_t)$ para $n=30$")
    plt.show()

    # Histograms of parameters
    burn4 = 2500
    burn30 = 1500
    plt.hist(alpha4_sims[burn4:],density=True,bins=100)
    plt.title(r"Histograma de frecuencia relativa de $\alpha$ para $n=4$")
    plt.show()
    plt.hist(beta4_sims[burn4:],density=True,bins=100,color="tab:red")
    plt.title(r"Histograma de frecuencia relativa de $\beta$ para $n=4$")
    plt.show()
    plt.hist(alpha30_sims[burn30:],density=True,bins=100,color="mediumseagreen")
    plt.title(r"Histograma de frecuencia relativa de $\alpha$ para $n=30$")
    plt.show()
    plt.hist(beta30_sims[burn30:],density=True,bins=100,color="tab:purple")
    plt.title(r"Histograma de frecuencia relativa de $\beta$ para $n=30$")
    plt.show()

    # Alternative distribution
    # n = 4
    alpha4alt_sims, beta4alt_sims = MHgamma_arc(data4,30000,sigma1=0.07)
    plt.plot(alpha4alt_sims,beta4alt_sims,alpha=0.75,marker='.',linewidth=0.5,markersize=0.75)
    plt.contour(X4,Y4,Z4,levels=50,cmap="viridis")
    plt.plot(alpha4alt_sims[0],beta4alt_sims[0],'ro')
    plt.title("Trayectoria de la cadena alternativa para la posterior con $n=4$")
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\alpha$")
    plt.show()

    # n = 30 
    alpha30alt_sims, beta30alt_sims = MHgamma_arc(data30,30000,sigma1=0.07)
    plt.plot(alpha30alt_sims,beta30alt_sims,alpha=0.75,marker='.',linewidth=0.5,markersize=0.75)
    plt.contour(X30,Y30,Z30,levels=50,cmap="viridis")
    plt.plot(alpha30alt_sims[0],beta30alt_sims[0],'ro')
    plt.title("Trayectoria de la cadena alternativa para la posterior con $n=30$")
    plt.ylabel(r"$\beta$")
    plt.xlabel(r"$\alpha$")
    plt.show()

    # Logarithm of density plots
    densgamma4alt = []
    for i in range(len(alpha4alt_sims)):
        densgamma4alt.append(poster_graf(alpha4alt_sims[i],beta4alt_sims[i],data4))
    plt.plot(np.log(densgamma4alt),linewidth=0.75)
    plt.title("Logaritmo de $f(X_t)$ alternativa para $n=4$")
    plt.show()
    
    densgamma30alt = []
    for i in range(len(alpha30alt_sims)):
        densgamma30alt.append(poster_graf(alpha30alt_sims[i],beta30alt_sims[i],data30))
    plt.plot(np.log(densgamma30alt),linewidth=0.75)
    plt.title("Logaritmo de $f(X_t)$ alternativa para $n=30$")
    plt.show()

    # Histograms of parameters for alternative
    burn4alt = 2500
    burn30alt = 1500
    plt.hist(alpha4alt_sims[burn4alt:],density=True,bins=100)
    plt.title(r"Histograma alternativo de frecuencia relativa de $\alpha$ para $n=4$")
    plt.show()
    plt.hist(beta4alt_sims[burn4alt:],density=True,bins=100,color="tab:red")
    plt.title(r"Histograma alternativo de frecuencia relativa de $\beta$ para $n=4$")
    plt.show()
    plt.hist(alpha30alt_sims[burn30alt:],density=True,bins=100,color="mediumseagreen")
    plt.title(r"Histograma alternativo de frecuencia relativa de $\alpha$ para $n=30$")
    plt.show()
    plt.hist(beta30alt_sims[burn30alt:],density=True,bins=100,color="tab:purple")
    plt.title(r"Histograma alternativo de frecuencia relativa de $\beta$ para $n=30$")
    plt.show()


    ###############################
    ######## Exercise 2 ###########
    ###############################

    gamma_sims = MHuga(10000,np.pi)
    plt.plot(gamma_sims)
    plt.title(r"Trayectoria de la cadena con propuesta $\Gamma([\alpha])$")
    plt.show()

    gamx = np.linspace(0,12,num=1000)
    gamy = gamma.pdf(gamx,np.pi,scale=1)
    plt.plot(gamx,gamy,label="Densidad teórica")
    plt.hist(gamma_sims[1:],density=True,bins=50,label="Histograma")
    plt.title(r"Histograma del muestreo con propuesta $\Gamma([\alpha])$")
    plt.legend()
    plt.show()

    gamma_traj = gamma.pdf(gamma_sims,np.pi,scale=1)
    plt.plot(gamma_traj)
    plt.xlabel("$t$")
    plt.ylabel("$f(X_t)$")
    plt.title("Gráfica de $f(X_t)$ contra $t$")
    plt.show()

    # Alternative
    gamma_alt_sims = MHuga_alt(10000,np.pi)
    plt.plot(gamma_sims)
    plt.title(r"Trayectoria de la cadena con propuesta $Unif(0,12)$")
    plt.show()
    
    plt.plot(gamx,gamy,label="Densidad teórica")
    plt.hist(gamma_alt_sims[1:],density=True,bins=50,label="Histograma")
    plt.title(r"Histograma del muestreo con propuesta $Unif(0,12))$")
    plt.legend()
    plt.show()


    ###############################
    ######## Exercise 3 ###########
    ###############################

    rw_sims1, rw_sims2 = rwmh(10000,1) # Error
    rw_sims_n1, rw_sims_n2 = rwmh(10000,0.5,starting_point=[1,1])

    # Evolution of chain
    plt.plot(rw_sims_n1, rw_sims_n2,alpha=0.9,marker='.',linewidth=0.3,markersize=0.5)
    plt.plot(rw_sims_n1[0],rw_sims_n2[0],'ro',label="Punto inicial")
    plt.title("Evolución de la cadena RWMH con propuesta normal")
    plt.legend()
    plt.show()

    # Logarithm of density plots
    mu = np.array([3,5],dtype=float)
    Sigma = np.array([[1, 0.9],[0.9, 1]], dtype=float)
    densrw = []
    for i in range(len(rw_sims_n1)):
        densrw.append((mvn.pdf([rw_sims_n1[i],rw_sims_n2[i]],mu,Sigma)))
    plt.plot(np.log(densrw[:1000]),linewidth=0.75)
    plt.title("Logaritmo de la densidad evaluada en la cadena para RWMH con propuesta normal")
    plt.show()

    # Histograms
    plt.hist(rw_sims_n1[100:],density=True,bins=50)
    plt.title(r"Histograma de la primera componente con propuesta $N(0,\sigma)$")
    plt.show()

    plt.hist(rw_sims_n2[100:],density=True,bins=50,color="mediumseagreen")
    plt.title(r"Histograma de la segunda componente con propuesta $N(0,\sigma)$")
    plt.show()


    ######## Alternative #########

    alt_rw_sims1, alt_rw_sims2 = rwmh_alt(10000,1)

    # Evolution of chain
    plt.plot(alt_rw_sims1, alt_rw_sims2,alpha=0.9,marker='.',linewidth=0.3,markersize=0.5)
    plt.plot(alt_rw_sims1[0],alt_rw_sims2[0],'ro',label="Punto inicial")
    plt.title("Evolución de la cadena RWMH con propuesta t de Student")
    plt.legend()
    plt.show()

    # Logarithm of density plots
    alt_densrw = []
    for i in range(len(alt_rw_sims1)):
        alt_densrw.append((mvn.pdf([alt_rw_sims1[i],alt_rw_sims2[i]],mu,Sigma)))
    plt.plot(np.log(alt_densrw[:1000]),linewidth=0.75)
    plt.title("Logaritmo de la densidad evaluada en la cadena para RWMH con propuesta t de Student")
    plt.show()

    # Histograms
    plt.hist(alt_rw_sims1[100:],density=True,bins=50)
    plt.title(r"Histograma de la primera componente con propuesta t de Student")
    plt.show()

    plt.hist(alt_rw_sims2[100:],density=True,bins=50,color="mediumseagreen")
    plt.title(r"Histograma de la segunda componente con propuesta t de Student")
    plt.show()