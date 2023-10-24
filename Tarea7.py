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
    n = np.max(np.shape(data))
    r1 = np.prod(data)
    r2 = np.sum(data)
    if (1 <= alpha <= 4) and (beta > 0):
        return (beta**(n*alpha) / Gamma(alpha)**n)*(r1**(alpha-1))*np.exp(-beta*(r2+1))
    else:
        return 0
if __name__ == "__main__":
    import numpy as np
    from scipy.special import gamma as Gamma
    from matplotlib import pyplot as plt
    
    data4 = np.random.gamma(shape=3,scale=(1/100),size=4)
    data30 = np.random.gamma(shape=3,scale=(1/100),size=30)
    delta = 0.01
    alpha_ax4 = np.arange(1,4,delta)
    beta_ax4 = np.arange(0,15,delta)
    X4, Y4 = np.meshgrid(alpha_ax4, beta_ax4)
    Z4 = poster_graf(X4,Y4,data4)
    plt.contour(X4,Y4,Z4,levels=500)
    plt.show()

    alpha_ax30 = np.arange(1,4,delta)
    beta_ax30 = np.arange(0,30,delta)
    X30, Y30 = np.meshgrid(alpha_ax30, beta_ax30)
    Z30 = poster_graf(X30,Y30,data30)
    plt.contour(X30,Y30,Z30,levels=500)
    plt.show()