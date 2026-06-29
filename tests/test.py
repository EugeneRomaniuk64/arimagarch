import ctypes 
import numpy as np
import timeit

ll_lib = ctypes.CDLL("test/log_likelihood.so")
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

ll_lib.log_likelihood.argtypes = [
    ctypes.POINTER(ctypes.c_double), # data[]
    ctypes.c_int, # data_len
    ctypes.c_int, # pad
    ctypes.POINTER(ctypes.c_int), # order[]
    ctypes.POINTER(ctypes.c_double), # params[]
    ctypes.POINTER(ctypes.c_double), # epsilon[]
    ctypes.POINTER(ctypes.c_double), # sigma2[]
    ctypes.c_bool # opmimized
]
ll_lib.log_likelihood.restype = ctypes.c_double

ll_lib.compute_gradient.argtypes = [
    ctypes.POINTER(ctypes.c_double), # data[]
    ctypes.c_int, # data_len
    ctypes.c_int, # pad
    ctypes.POINTER(ctypes.c_int), # order[]
    ctypes.POINTER(ctypes.c_double), # params[]
    ctypes.POINTER(ctypes.c_double), # epsilon[]
    ctypes.POINTER(ctypes.c_double), # sigma2[]
    ctypes.POINTER(ctypes.c_double), # d_eps[]
    ctypes.POINTER(ctypes.c_double), # d_sig2[]
    ctypes.POINTER(ctypes.c_double) # grad[]
]

ll_lib.compute_gradient.restype = None

optimized = True

res = ll_lib.log_likelihood(
    y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(y_pad_len),
    ctypes.c_int(pad),
    order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_bool(optimized)
)

print(type(res))

d_eps = np.zeros((y_pad_len, 2 + sum(order)), dtype=np.float64)
d_sig2 = np.zeros((y_pad_len, 2 + sum(order)), dtype=np.float64)
grad = np.zeros( 2 + sum(order), dtype=np.float64)

ll_lib.compute_gradient(
    y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int(y_pad_len),
    ctypes.c_int(pad),
    order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    d_eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    d_sig2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

print(grad)


time = timeit.timeit(
    lambda: ll_lib.log_likelihood(
        y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(y_pad_len),
        ctypes.c_int(pad),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_bool(optimized)
    ), 
    number=100000
)
print(f"average: {time / 100000 * 1000:.4f} ms per call")


def neg_ll(params):
    grad[:] = 0
    return -ll_lib.log_likelihood(
        y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(y_pad_len),
        ctypes.c_int(pad),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_bool(False)
    )

def grad_vec(params):
    epsilon[:] = 0
    sigma2[:pad] = y_var
    sigma2[pad:] = 0
    d_eps[:] = 0
    d_sig2[:] = 0
    grad[:] = 0
    
    ll_lib.log_likelihood(
        y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(y_pad_len),
        ctypes.c_int(pad),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_bool(False)
    )
    
    ll_lib.compute_gradient(
        y_pad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(y_pad_len),
        ctypes.c_int(pad),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        params.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        d_eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        d_sig2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    return -grad.copy()

from scipy.optimize import approx_fprime

n, m, p, q = order

x0 = np.concatenate([
    [y_mean],
    [y_var * 0.1],
    np.zeros(n),
    np.zeros(m),
    np.full(q, 0.05),
    np.full(p, 0.5)
])

numerical = approx_fprime(x0, neg_ll, 1e-7)
analytical = grad_vec(x0)

param_names = ['mu', 'omega'] + \
              [f'phi_{i}' for i in range(order[0])] + \
              [f'theta_{i}' for i in range(order[1])] + \
              [f'alpha_{i}' for i in range(order[2])] + \
              [f'beta_{i}' for i in range(order[3])]

for name, n, a in zip(param_names, numerical, analytical):
    rel_err = abs(n - a) / (abs(n) + 1e-10)
    print(f"{name:12s}  numerical: {n:12.6f}  analytical: {a:12.6f}  rel_err: {rel_err:.2e}")


from scipy.optimize import minimize

def ll_wrapper(params):
    return neg_ll(params), grad_vec(params)



bnds = (
    [(None, None)] +
    [(1e-6, None)] +
    [(None, None)] * 1 +
    [(None, None)] * 2 +
    [(1e-6, 0.999)] * 1 +
    [(1e-6, 0.999)] * 1
)

res = minimize(
    fun=ll_wrapper,
    x0=x0,
    method='L-BFGS-B',
    jac=True,
    bounds=bnds
)

print(res)

print(res.x)

from scipy.optimize import check_grad
print(check_grad(neg_ll, grad_vec, params, epsilon=1e-7))
# epsilon = np.zeros(len(y_pad)) #Used to be len(y) + pa
# sigma2 = np.full(len(y) + pad, y_var)

# time = timeit.timeit(
#     lambda: _get_loglikelihood(params, y_pad, y_mean, y_var, pad, order, epsilon, sigma2), 
#     number=100000
# )
# print(f"average: {time / 100000 * 1000:.4f} ms per call")


# print(_get_loglikelihood(params, y_pad, y_mean, y_var, pad, order, epsilon, sigma2))

