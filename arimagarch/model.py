import numpy as np
import scipy.optimize
import arch.univariate
import time

LOG_2PI = np.log(np.pi * 2)

def loglikelihood(params, y_pad, n, m, p, q, y_mean, y_var, pad):
    """Log-likelood computation for ARMA(n,m)-GARCH(p,q) model

    Args:
        params (tuple): model parameters
        y_pad (ndarray): padded data
        n (int): the order of the AR term
        m (int): the order of the MA term
        p (int): the order of the GARCH term sigma^2
        q (int): the order of the ARCH term epsilon^2
        y_mean (float64): the mean of y
        y_var (float64): the variance of y
        pad (int): the padding of y

    Returns:
        float64: negative log-likehood for the data
    """
    mu = params[0]
    omega = params[1]
    
    phis = params[2 : 2+n]
    thetas = params[2+n : 2+m+n]
    alphas = params[2+m+n : 2+m+n+q]
    betas = params[2+m+n+q : 2+m+n+q+p]
    
    #TODO Change the initial values
    epsilon = np.zeros(len(y) + pad)
    epsilon[pad] = y_pad[pad] - y_mean
    sigma2 = np.full(len(y) + pad, y_var)
    loglikelihood = 0

    for i in range(pad+1, len(y_pad)):
        # epsilon[i] = y_pad[i] - mu - phis @ y_pad[i-1 : i-1-n : -1] - thetas @ epsilon[i-1 : i-1-m : -1]
        # sigma2[i] = omega + alphas @ epsilon[i-1 : i-1-q : -1]**2 + betas @ sigma2[i-1 : i-1-p : -1]
        
        epsilon[i] = y_pad[i] - mu
        for j in range(n):
            epsilon[i] -= phis[j] * y_pad[i-j-1]
        
        for j in range(m):
            epsilon[i] -= thetas[j] * epsilon[i-j-1] 
        
        sigma2[i] = omega
        for j in range(q):
            sigma2[i] += alphas[j] * epsilon[i-j-1]**2
        
        for j in range(p):
            sigma2[i] += betas[j] * sigma2[i-j-1]
            
    loglikelihood = -0.5 * np.sum(LOG_2PI + np.log(sigma2[1:]) + epsilon[1:]**2 / sigma2[1:])
    
    return -loglikelihood


def fit(y, n, m, p, q):
    """Fits an AR(n)-GARCH(1,1) model

    Args:
        y (ndarray): data
        n (int): the order of the AR term
        m (int): the order of the MA term
        p (int): the order of the GARCH term sigma^2
        q (int): the order of the ARCH term epsilon^2

    Returns:
        touple[ndarray, float64]: a touple containing the estimated coefficients and the maximized log-likelyhood value
    """
    bnds = [(None, None), (1e-6, 1)]
    
    for i in range(n):
        bnds.append((None, None))
    
    for i in range(m):
        bnds.append((None, None))
    
    for i in range(q):
        bnds.append((0, None))
    
    for i in range(p):
        bnds.append((0, None))
        

    alpha_idx = list(range(2+n+m, 2+n+m+q))
    beta_idx = list(range(2+n+m+q, 2+n+m+q+p))        
        
    constr = [{'type': 'ineq', 'fun': lambda x: 1 - sum(x[i] for i in alpha_idx) - sum(x[i] for i in beta_idx)}] 
    
    pad = max(n, m, p, q)
    y_pad = np.concatenate([np.zeros(pad), y])
    
    y_mean = np.mean(y)
    y_var = np.var(y)
    print(type(pad))
    
    x0 = np.concatenate([
    [y_mean],
    [y_var * 0.1],
    np.zeros(n),
    np.zeros(m),
    np.full(q, 0.1),
    np.full(p, 0.8)
    ])
    
    res = scipy.optimize.minimize(
        fun=loglikelihood,
        x0=x0,
        args=(y_pad, n, m, p, q, y_mean, y_var, pad),
        bounds=bnds,
        constraints=constr,
        method='SLSQP',
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    return res.x, -res.fun
    
y = np.array([0.1, 0.2, -0.1, 0.2, 0.1, 0.02, -0.02, 0.1, -0.2, 0.1, 0.02, 0.03, -0.01, -0.02, 0.004, 0.02, -0.04, -0.1, 0.1])

n = 1
m = 0
p = 1
q = 1



pad = max(n, m, p, q)
y_pad = np.concatenate([np.zeros(pad), y])

#print(loglikelihood(x0, y_pad, n, m, p, q, y_mean=np.mean(y), y_var=np.var(y), pad=pad))



params, res = fit(y, n, m, p, q)


# model = arch.univariate.ARX(y, lags=n)
# model.volatility = arch.univariate.GARCH(p,0,q)
# model.distribution = arch.univariate.Normal()



# print(model.fit())

print(f'''Minimized log-likelyhood with the following coefficients: {[float(params[i]) for i in range(len(params))]}\nLog-likehood value: {res:,.3f}
    ''')
