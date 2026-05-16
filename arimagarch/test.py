import ctypes 
import numpy as np
import timeit

ll = ctypes.CDLL("arimagarch/log_likelihood.so")
LOG_2PI = np.log(np.pi * 2)

def simulate(T: int, params=None, order=tuple[int], seed:float = 10): 
    np.random.seed(seed)
    
    n, m, p, q = order
    mu    = params[0]
    omega = params[1]
    phis   = params[2:2+n]
    thetas = params[2+n:2+n+m]
    alphas = params[2+n+m:2+n+m+q]
    betas  = params[2+n+m+q:2+n+m+q+p]

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


def _get_loglikelihood(params, y_pad, y_mean, y_var, pad, order, epsilon, sigma2):
        """Log-likelood computation for ARMA(n,m)-GARCH(p,q) model

        Args:
            params (tuple): model parameters
            y_pad (ndarray): padded data
            y_mean (float64): the mean of y
            y_var (float64): the variance of y
            pad (int): the padding of y

        Returns:
            float64: negative log-likehood for the data
        """
        n, m, p, q = order
        
        mu = params[0]
        omega = params[1]
        
        phis = params[2 : 2+n]
        thetas = params[2+n : 2+m+n]
        alphas = params[2+m+n : 2+m+n+q]
        betas = params[2+m+n+q : 2+m+n+q+p]
        
        # if omega <= 0:
        #     return 1e15
        # if np.any(alphas < 0) or np.any(betas < 0):
        #     return 1e15
        # if np.sum(alphas) + np.sum(betas) >= 1:
        #     return 1e15
        
        #TODO Change the initial values
        
        loglikelihood = 0
        
        for i in range(pad, len(y_pad)):
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
                
                
        loglikelihood = -0.5 * np.sum(LOG_2PI + np.log(sigma2[pad:]) + epsilon[pad:]**2 / sigma2[pad:]) #TODO: Make sure it's pad + 1 and not pad
        
        return -loglikelihood


order = np.array([1,2,1,1], dtype=np.int32)
params = np.array([0.2, 0.5, 0.2, 0.1, 0.05, 0.4, 0.3], dtype=np.float64)

y = simulate(1000, params=params, order=order.astype(int))
assert(y.dtype == np.float64)

pad = max(order)
y_pad = np.concatenate((np.zeros(pad), y))
y_pad_len = len(y_pad)

y_mean = np.mean(y)
y_var = np.var(y)

epsilon = np.zeros(y_pad_len, dtype=np.float64)

sigma2_pad = np.full(pad, y_var, dtype=np.float64)
sigma2 = np.concatenate((sigma2_pad, np.zeros(len(y))), dtype=np.float64)

ll.log_likelihood.argtypes = [
    ctypes.POINTER(ctypes.c_double), # data[]
    ctypes.c_int, # data_len
    ctypes.c_int, # pad
    ctypes.POINTER(ctypes.c_double), # params[]
    ctypes.POINTER(ctypes.c_int), # order[]
    ctypes.POINTER(ctypes.c_double), # epsilon[]
    ctypes.POINTER(ctypes.c_double), # sigma2[]
    ctypes.c_bool # omimized
]

ll.log_likelihood.restype = ctypes.c_double

optimized = True

res = ll.log_likelihood(
    y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(y_pad_len),
    ctypes.c_int(pad),
    params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_bool(optimized)
)

print(res)


time = timeit.timeit(
    lambda: ll.log_likelihood(
        y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(y_pad_len),
        ctypes.c_int(pad),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_bool(optimized)
    ), 
    number=1000000
)
print(f"average: {time / 1000000 * 1000:.4f} ms per call")

print(f"key: {order[0]*1000 + order[1]*100 + order[2]*10 + order[3]}")

# epsilon = np.zeros(len(y_pad)) #Used to be len(y) + pa
# sigma2 = np.full(len(y) + pad, y_var)

# time = timeit.timeit(
#     lambda: _get_loglikelihood(params, y_pad, y_mean, y_var, pad, order, epsilon, sigma2), 
#     number=100000
# )
# print(f"average: {time / 100000 * 1000:.4f} ms per call")


# print(_get_loglikelihood(params, y_pad, y_mean, y_var, pad, order, epsilon, sigma2))

