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

def MH_beta(iterations,trials,successes):
    '''This function implements the Metropolis-Hastings algorithm to simulate from 
    the posterior of the parameter in Bayesian inference'''
    import numpy as np
    # Starting point uniform on (0,1/2)
    current_point = np.random.random()*0.5
    output = []
    output.append(current_point)
    for i in range(iterations):
        proposal = np.random.beta(successes+1,trials-successes+1)
        rho = rho_beta(current_point,proposal)
        u = np.random.random()
        if u <= rho:
            current_point = proposal
        output.append(current_point)
    return output
    

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
    for i in range(iterations):
        proposal = np.random.random()*0.5
        rho = rho_unif(current_point,proposal,trials,successes)
        u = np.random.random()
        if u <= rho:
            current_point = proposal
        output.append(current_point)
    return output
    
def log_post(p,n,r):
    return np.log(posterior_prop(p,n,r))

# Examples and homework
if  __name__ == "__main__":
    import numpy as np
    np.random.seed(57)
   
    # For LaTeX graphs
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.style.use("seaborn-v0_8")
    plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,})

    # Simple example of MH_beta()
    simple_sample = MH_beta(1000,50,28)
    plt.hist(simple_sample,density=True)
    plt.title("Histograma con propuesta beta y $n=50, r=28$")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.savefig("tarea6/explot.pgf")
    
    # Samples of Bernoulli
    sample5 = np.random.binomial(5,1/3,1)
    sample40 = np.random.binomial(40,1/3,1)
    r5 = np.sum(sample5)
    r40 = np.sum(sample40)

    # Implementation with data
    MH_beta_sims5 = MH_beta(1000,5,r5)
    MH_beta_sims40 = MH_beta(1000,40,r40)
    
    # Plots of sampling process
    fig_beta5, axb5 = plt.subplots(figsize=(7,4))
    fig_beta40, axb40 = plt.subplots(figsize=(7,4))

    axb5.plot(MH_beta_sims5)
    axb40.plot(MH_beta_sims40)

    fig_beta5.suptitle(r"Muestreo con propuesta beta y $n=5$")
    fig_beta40.suptitle(r"Muestreo con propuesta beta y $n=40$")
    axb5.set_xlabel(r"Paso de la cadena ($t$)")
    axb5.set_ylabel(r"Valor de $X_t$")
    axb40.set_xlabel(r"Paso de la cadena ($t$)")
    axb40.set_ylabel(r"Valor de $X_t$")

    # Histograms
    xpoints = np.linspace(0,0.5,num=100)
    figdens5, axdens5 = plt.subplots(figsize=(7,4))
    figdens40, axdens40 = plt.subplots(figsize=(7,4))

    dens5 = []
    for point in xpoints:
        dens5.append(posterior_prop(point,5,r5))
    
    dens40 = []
    for point in xpoints:
        dens40.append(posterior_prop(point,40,r40))

    axdens5.plot(xpoints,dens5)
    axdens40.plot(xpoints,dens40)

    figdens5.suptitle(r"Múltiplo de la densidad objetivo con $n=5, r=$" + str(r5))
    figdens40.suptitle(r"Múltiplo de la densidad objetivo con $n=40, r=$" + str(r40))

    hist_beta5, abh5 = plt.subplots(figsize=(7,4))
    hist_beta40, abh40 = plt.subplots(figsize=(7,4))

    abh5.hist(MH_beta_sims5,density=True,alpha=0.8)
    abh40.hist(MH_beta_sims40,density=True,alpha=0.8)
    abh5.set_xlabel("Valor")
    abh5.set_ylabel("Frecuencia")
    abh40.set_xlabel("Valor")
    abh40.set_ylabel("Frecuencia")
    hist_beta40.suptitle("Histograma para la muestra con propuesta beta y $n=40, r=$" + str(r40))
    hist_beta5.suptitle("Histograma para la muestra con propuesta beta y $n=5, r=$" + str(r5))

    # Implementation with the new proposal
    MH_unif_sims5 = MH_unif(1000,5,r5)
    MH_unif_sims40 = MH_unif(1000,40,r40)

    fig_unif40, axu40 = plt.subplots(figsize=(7,4))
    fig_unif5, axu5 = plt.subplots(figsize=(7,4))

    axu5.plot(MH_unif_sims5)
    axu40.plot(MH_unif_sims40)

    fig_unif5.suptitle(r"Muestreo con propuesta $U(01)$ y $n=5$")
    fig_unif40.suptitle(r"Muestreo con propuesta $U(0,1)$ y $n=40$")
    axu5.set_xlabel(r"Paso de la cadena ($t$)")
    axu5.set_ylabel(r"Valor de $X_t$")
    axu40.set_xlabel(r"Paso de la cadena ($t$)")
    axu40.set_ylabel(r"Valor de $X_t$")

    # Histograms
    hist_unif5, abu5 = plt.subplots(figsize=(7,4))
    hist_unif40, abu40 = plt.subplots(figsize=(7,4))

    abu5.hist(MH_unif_sims5,density=True,color="#2ca02c",alpha=0.5)
    abu40.hist(MH_unif_sims40,density=True,color="#2ca02c",alpha=0.5)
    hist_unif40.suptitle("Histograma para la muestra con propuesta $U(0,1)$ y $n=40, r=$" + str(r40))
    hist_unif5.suptitle("Histograma para la muestra con propuesta $U(0,1)$ y $n=5, r=$" + str(r5))
    abu5.set_xlabel("Valor")
    abu5.set_ylabel("Frecuencia")
    abu40.set_xlabel("Valor")
    abu40.set_ylabel("Frecuencia")

    # Save figures
    fig_beta5.savefig("tarea6/fig_beta5.pgf")
    fig_beta40.savefig("tarea6/fig_beta40.pgf")
    fig_unif5.savefig("tarea6/fig_unif5.pgf")
    fig_unif40.savefig("tarea6/fig_unif40.pgf")
    figdens5.savefig("tarea6/fig_dens5.pgf")
    figdens40.savefig("tarea6/fig_dens40.pgf")
    hist_beta5.savefig("tarea6/hist_beta5.pgf")
    hist_beta40.savefig("tarea6/hist_beta40.pgf")
    hist_unif5.savefig("tarea6/hist_unif5.pgf")
    hist_unif40.savefig("tarea6/hist_unif40.pgf")

    # Log density for burn-in
    logdensb5 = []
    for i in MH_beta_sims5:
        logdensb5.append(log_post(i,5,r5))
    logdensb40 = []
    for i in MH_beta_sims40:
        logdensb40.append(log_post(i,40,r40))
    logdensu5 = []
    for i in MH_unif_sims5:
        logdensu5.append(log_post(i,5,r5))
    logdensu40 = []
    for i in MH_unif_sims40:
        logdensu40.append(log_post(i,40,r40))

    figlogb5, alb5 = plt.subplots(figsize=(7,4))
    figlogb40, alb40 = plt.subplots(figsize=(7,4))
    figlogu5, alu5 = plt.subplots(figsize=(7,4))
    figlogu40, alu40 = plt.subplots(figsize=(7,4))

    alb5.plot(logdensb5)
    alb40.plot(logdensb40)
    alu5.plot(logdensu5)
    alu40.plot(logdensu40)

    figlogb5.suptitle("Gráfica de $\log f(X_t)$ para propuesta beta, $n=5, r=$"+str(r5))
    figlogb40.suptitle("Gráfica de $\log f(X_t)$ para propuesta beta, $n=40, r=$"+str(r40))
    figlogu5.suptitle("Gráfica de $\log f(X_t)$ para propuesta $U(0,1)$, $n=5, r=$"+str(r5))
    figlogu40.suptitle("Gráfica de $\log f(X_t)$ para propuesta $U(0,1)$, $n=40, r=$"+str(r40))

    figlogb5.savefig("tarea6/figlogb5.pgf")
    figlogb40.savefig("tarea6/figlogb40.pgf")
    figlogu5.savefig("tarea6/figlogu5.pgf")
    figlogu40.savefig("tarea6/figlogu40.pgf")