from arimagarch import simulate, ARMAGARCH


n = 1
m = 1
p = 1
q = 1

true_params = [0.1, 0.05, 0.3, 0.2, 0.1, 0.7]

y = simulate(1000, order=(n,m,p,q), params=true_params)

import timeit

model = ARMAGARCH(y, order=(n,m,p,q))

res = model.fit(optimized=True, grad_method='analytic')
print(res.summary())

time = timeit.timeit(lambda: model.fit(), number=1000)
print(f"M1. average: {time / 1000 * 1000:.4f} ms per call")


# import arch

# model = arch.univariate.ARX(y, lags=n)
# model.volatility = arch.univariate.GARCH(p,0,q)
# model.distribution = arch.univariate.Normal()

# res = model.fit()
# print(res.summary())

# time = timeit.timeit(lambda: model.fit(), number=1000)
# print(f"M2. average: {time / 1000 * 1000:.4f} ms per call")