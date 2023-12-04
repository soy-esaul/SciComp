# Funciones

# Problema 1

def logfac(n):
    n = int(n)
    result = 0
    for i in range(n):
        result += log(i+1)
    return result

def ecologpdf(x,N,p):
    alpha = 1
    beta  = 20
    m = np.shape(x)[0]
    logfx = np.array([logfac(int(i)) for i in x],dtype=float)
    logfNx = np.array([logfac(int(N-j)) for j in x],dtype=float)
    return (m*logfac(int(N))-sum(logfx)-sum(logfNx)
        + (alpha-1+sum(x))*log(p)+(beta-1+m*N-sum(x))*log(1-p))

def MHeco(iterations=100):
    alpha = 1
    beta  = 20
    x = np.array([7,7,8,8,9,4,7,5,5,6,9,8,11,7,5,5,7,3,10,3],dtype=float)
    p = []
    N = []
    m = int(np.shape(x)[0])
    Nmax = 1000
    current_p = uniform.rvs()
    current_N = randint.rvs(low=np.max(x),high=Nmax)
    p.append(current_p)
    N.append(current_N)
    for t in range(iterations):
        unif = uniform.rvs()
        if unif < 0.2:
            if m*current_N-sum(x)+beta > 0:
                proposal = beta_var.rvs(a=sum(x)+alpha,b=m*current_N-sum(x)+beta)
                current_p = proposal
            else:
                pass
        elif unif >= 0.2 and unif < 0.4:
            proposal_p = beta_var.rvs(a=alpha,b=beta)
            proposal_N = randint.rvs(low=0,high=Nmax)
            V  = ecologpdf(x,current_N,current_p)
            Vp = ecologpdf(x,proposal_N,proposal_p)
            W  = beta_var.logpdf(current_p,a=alpha,b=beta)
            Wp = beta_var.logpdf(proposal_p,a=alpha,b=beta)
            unif = np.random.random()
            if log(unif) < Vp - V - Wp + W and proposal_N < Nmax:
                current_p = proposal_p
                current_N = proposal_N
        elif unif >= 0.4 and unif < 0.6:
            M = int(Nmax*current_p)+1
            proposal_N = hypergeom.rvs(Nmax,500,M)
            V  = ecologpdf(x,current_N,current_p)
            Vp = ecologpdf(x,proposal_N,current_p)
            W  = hypergeom.logpmf(current_N,Nmax,500,int(proposal_N))
            Wp = hypergeom.logpmf(proposal_N,Nmax,500,int(current_N))
            unif = np.random.random()
            if log(unif) < Vp - V + W - Wp and proposal_N < Nmax:
                current_N = proposal_N
        elif unif >= 0.6 and unif < 0.8:
            rate = int(current_N)
            proposal_N = np.max(x) + poisson.rvs(rate)
            rate_p = int(proposal_N)
            V  = ecologpdf(x,current_N,current_p)
            Vp = ecologpdf(x,proposal_N,current_p)
            W  = poisson.logpmf(current_N,mu=rate)
            Wp = poisson.logpmf(proposal_N,mu=rate_p)
            unif = np.random.random()
            if log(unif) < Vp - V + W - Wp and proposal_N < Nmax:
                current_N = proposal_N
        elif unif >= 0.8 and unif < 0.825:
            proposal_N = binom.rvs(n=Nmax,p=current_p)
            V  = ecologpdf(x,current_N,current_p)
            Vp = ecologpdf(x,proposal_N,current_p)
            W  = binom.logpmf(current_N,n=Nmax,p=current_p)
            Wp = binom.logpmf(proposal_N,n=Nmax,p=current_p)
            unif = np.random.random()
            if log(unif) < Vp - V + W - Wp and proposal_N < Nmax:
                current_N = proposal_N
        else:
            rule = np.random.random()
            epsilon = int(1*(rule < 0.5) - 1*(rule >= 0.5))
            proposal_N = current_N + epsilon
            V  = ecologpdf(x,current_N,current_p)
            Vp = ecologpdf(x,proposal_N,current_p)
            unif = np.random.random()
            if log(unif) < Vp - V and proposal_N < Nmax:
                current_N = proposal_N
        p.append(current_p)
        N.append(current_N)
    p = np.array(p,dtype=float)
    N = np.array(N,dtype=float)
    return p, N
    

# Problema 2

def logmerc(a,b,c,x,y):
    quo =  (x-a)**2 / 2*b**2
    lfy = np.array([logfac(i) for i in y],dtype=float)
    return (log(c)*sum(y) - sum(y*quo) - sum(lfy) - c*sum(exp(-quo))
            -(((a-35)**2)/(50)) + loggammadens(c,3,3/950) 
            + loggammadens(b,2,2/5))

def loggammadens(x,alpha,beta):
    return (-log(gamma(alpha)) + alpha*log(beta) + (alpha-1)*log(x)
            -beta*x)

def MHmerc(iterations=100,starting_point=[25,3,5000]):
    starting_point = np.array(starting_point,dtype=float)
    output = []
    current_point = starting_point
    for t in range(iterations):
        a = current_point[0]
        b = current_point[1]
        c = current_point[2]
        unif = np.random.random()
        if unif < 0.333:
            a_p = gamma_var.rvs(a=54,scale=1/2)
            V = logmerc(a,b,c,X,Y)
            Vp = logmerc(a_p,b,c,X,Y)
            unif = np.random.random()
            if log(unif) < min(Vp - V,0):
                current_point[0] = a_p
            current_point[0] = a_p
        elif unif >= 0.333 and unif < 0.666:
            b_p = gamma_var.rvs(a=3,scale=1/2)
            V   = logmerc(a,b,c,X,Y)
            Vp  = logmerc(a,b_p,c,X,Y)
            unif = np.random.random()
            if log(unif) < min(Vp - V,0):
                current_point[1] = b_p
            current_point[1] = b_p
        elif unif >=0.666 and unif < 0.999:
            epsilon = normal.rvs(loc=0,scale=200)
            c_p = c + epsilon
            V   = logmerc(a,b,c,X,Y)
            Vp  = logmerc(a,b,c_p,X,Y)
            unif = np.random.random()
            if log(unif) < Vp - V:
                current_point[2] = c_p
        else:
            pass
        output.append(current_point.copy())
    output = np.array(output,dtype=float)
    return output 







# Ejercicios
if __name__ == "__main__":
    import numpy as np
    from scipy.stats import uniform
    from scipy.stats import randint
    from scipy.stats import norm as normal
    from scipy.stats import beta as beta_var
    from scipy.stats import hypergeom
    from scipy.stats import poisson
    from scipy.stats import gamma as gamma_var
    from scipy.stats import binom 
    from scipy.special import gamma 
    from numpy import pi
    from numpy import log
    from numpy import exp
    from scipy.special  import factorial

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use("seaborn-v0_8")

    np.random.seed(57)

    # Ejercicio 1

    eco = MHeco(iterations=10000)
    p, N = eco
    plt.plot(p)
    plt.title("Trayectoria para el muestreo de p")
    plt.xlabel("Iteraciones")
    plt.ylabel(r"$p$")
    plt.savefig("Tarea9/trajp.png")
    plt.show()

    plt.plot(N)
    plt.title("Trayectoria para el muestreo de N")
    plt.xlabel("Iteraciones")
    plt.ylabel(r"$N$")
    plt.savefig("Tarea9/trajN.png")
    plt.show()

    plt.plot(p,N)
    plt.title("Trayectoria de la cadena para el muestreo poblacional")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$p$")
    plt.savefig("Tarea9/trajpN.png")
    plt.show()

    # Histogramas
    plt.hist(p[100:],density=True,alpha=0.8,bins=20)
    plt.title(r"Histograma para el muestreo de $p$")
    plt.savefig("Tarea9/histp.png")
    plt.show()
    plt.hist(N,density=True,alpha=0.8,bins=20)
    plt.title(r"Histograma para el muestreo de $N$")
    plt.savefig("Tarea9/histN.png")
    plt.show()

    # Burn-in
    logobj = []
    for i in range(np.shape(p)[0]):
        logobj.append(ecologpdf(x,N[i],p[i]))

    plt.plot(logobj)
    plt.title("Gráfica de burn-in para el ejercicio 1")
    plt.xlabel("Iteraciones")
    plt.ylabel(r"$\log(f(X_t))"$)
    plt.savefig("Tarea9/burnin1.png")
    plt.show() 
    



    # Ejercicio 2

    X = np.array([ 25, 18, 19, 51, 16, 59, 16, 54, 52, 16, 31, 31, 54, 26, 19, 13, 59, 48, 54, 23, 50, 59,
    55, 37, 61, 53, 56, 31, 34, 15, 41, 14, 13, 13, 32, 46, 17, 52, 54, 25, 61, 15, 53, 39, 33, 52, 65,
    35, 65, 26, 54, 16, 47, 14, 42, 47, 48, 25, 15, 46, 31, 50, 42, 23, 17, 47, 32, 65, 45, 28, 12, 22,
    30, 36, 33, 16, 39, 50, 13, 23, 50, 34, 19, 46, 43, 56, 52,42, 48, 55, 37, 21, 45, 64, 53, 16, 62,
    16, 25, 62])

    Y = np.array([1275, 325, 517, 0, 86, 0, 101, 0, 0, 89, 78, 83, 0, 1074, 508, 5, 0, 0, 0, 1447, 0, 0,
    0, 0, 0, 0, 0, 87, 7, 37, 0, 15, 5, 6, 35, 0, 158, 0, 0, 1349, 0, 35, 0, 0, 12, 0, 0, 2, 0, 1117, 0,
    79, 0, 13, 0, 0, 0, 1334, 56, 0, 81, 0, 0, 1480, 177, 0, 29, 0, 0, 551, 0, 1338, 196, 0, 9, 104, 0,
    0, 3, 1430, 0, 2, 492, 0, 0, 0, 0, 0, 0, 0, 0, 1057, 0, 0, 0, 68, 0, 87, 1362, 0])

    # Muestrear los datos
    mercado = MHmerc(iterations=10000)

    # Figuras de la cadena
    plt.plot(mercado[:,0])
    plt.xlabel("Iteraciones")
    plt.ylabel("Edad (años)")
    plt.title("Trayectoria para el muestreo de la edad en años")
    plt.savefig("Tarea9/traja.png")
    plt.show()
    plt.plot(mercado[:,1])
    plt.xlabel("Iteraciones")
    plt.ylabel("Amplitud del segmento (años)")
    plt.title("Trayectoria para el muestreo de la amplitud en años")
    plt.savefig("Tarea9/trajb.png")
    plt.show()
    plt.plot(mercado[:,2])
    plt.xlabel("Iteraciones")
    plt.ylabel("Gasto promedio")
    plt.title("Trayectoria para el muestreo del gasto promedio")
    plt.savefig("Tarea9/trajc.png")
    plt.show()

    # Histogramas
    plt.hist(mercado[:,0],density=True,alpha=0.8,bins=50)
    plt.title("Histograma para el muestreo de la edad en años")
    plt.savefig("Tarea9/hista.png")
    plt.show()
    plt.hist(mercado[:,1],density=True,alpha=0.8,bins=50)
    plt.title("Histograma para el muestreo de la amplitud en años")
    plt.savefig("Tarea9/histb.png")
    plt.show()
    plt.hist(mercado[:,2],density=True,alpha=0.8,bins=50)
    plt.title("Histograma para el muestreo del gasto promedio")
    plt.savefig("Tarea9/histc.png")
    plt.show()