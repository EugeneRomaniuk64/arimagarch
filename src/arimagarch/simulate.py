import numpy as np

def simulate(T, params=None, order=tuple[int], seed:float = 10): 
    np.random.seed(seed)
    
    n, m, p, q = order
    mu = params[0]
    omega = params[1]
    phis = params[2:2+n]
    thetas = params[2+n:2+n+m]
    alphas = params[2+n+m:2+n+m+q]
    betas = params[2+n+m+q:2+n+m+q+p]

    pad = max(n, m, p, q)
    T_total = T + pad

    y = np.zeros(T_total)
    epsilon = np.zeros(T_total)
    sigma2 = np.zeros(T_total)


    uncond_var = omega / (1 - np.sum(alphas) - np.sum(betas))
    sigma2[:pad+1] = uncond_var

    for i in range(pad, T_total):
        sigma2[i] = omega
        for j in range(q):
            sigma2[i] += alphas[j] * epsilon[i-j-1]**2
        for j in range(p):
            sigma2[i] += betas[j] * sigma2[i-j-1]

  
        epsilon[i] = np.random.normal(0, np.sqrt(sigma2[i]))

  
        y[i] = mu + epsilon[i]
        for j in range(n):
            y[i] += phis[j] * y[i-j-1]
        for j in range(m):
            y[i] += thetas[j] * epsilon[i-j-1]

    return y[pad:]  