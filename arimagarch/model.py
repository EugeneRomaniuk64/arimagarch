import numpy as np
import numdifftools as nd
import scipy.optimize
import warnings
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from datetime import datetime as dt
from functools import cached_property
from collections.abc import Callable
from numpy import ndarray


LOG_2PI = np.log(np.pi * 2)

def format_float(number: float) -> str:
    return f"{number:.3f}"
    
def simulate(T, params=None, order=tuple[int], seed:float = 10): 
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

class ARMAGARCH:
    def __init__(self, data, order: tuple[int]):
        self.y = data
        self.order = order
        self._dep_var_name = None #TODO: Add var name for pandas
        self.num_params = sum(order) + 2 # add 2 for mu and omega
    
    def _get_loglikelihood(self, params, y_pad, y_mean, y_var, pad):
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
        n, m, p, q = self.order
        
        mu = params[0]
        omega = params[1]
        
        phis = params[2 : 2+n]
        thetas = params[2+n : 2+m+n]
        alphas = params[2+m+n : 2+m+n+q]
        betas = params[2+m+n+q : 2+m+n+q+p]
        
        if omega <= 0:
            return 1e15
        if np.any(alphas < 0) or np.any(betas < 0):
            return 1e15
        if np.sum(alphas) + np.sum(betas) >= 1:
            return 1e15
        
        #TODO Change the initial values
        epsilon = np.zeros(len(y_pad)) #Used to be len(y) + pad
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
                
                
        loglikelihood = -0.5 * np.sum(LOG_2PI + np.log(sigma2[pad+1:]) + epsilon[pad+1:]**2 / sigma2[pad+1:]) #TODO: Make sure it's pad + 1 and not pad

            
        return -loglikelihood
    
        
    def _get_cov(self, ll_fixed: Callable) -> ndarray:
        """Computes the covariance matrix for the log likehood function

        Raises:
            ValueError: Variance cannon be negative

        Returns:
            np.ndarray: an array representing the covariance matrix
        """
        
        try:
            
            H = nd.Hessian(ll_fixed)(self.params_est)
            cov = np.linalg.pinv(H)
            #print(f"Variance{np.diag(cov)}")
            if np.any(np.diag(cov) < 0):
                raise ValueError("Non-positive variance estimates")
            
            return cov
        except Exception as e:
            warnings.warn(f"Could not compute covariance matrix: {e}")
            return np.full((len(self.params_est), len(self.params_est)), np.nan)
        

    def fit(self):
        """Fits an ARMA(n,m)-GARCH(p,q) model

        Returns:
            ModelResults: an object containing the results on the model
        """
        
        n, m, p, q = self.order
        
        bnds = [(None, None), (1e-6, None)] #TODO Make sure all the bounds are correct
        
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
            
        constr = [{'type': 'ineq', 'fun': lambda x: 1 - sum(x[i] for i in alpha_idx) - sum(x[i] for i in beta_idx) - 1e-6}] 
        
        ar_idx = list(range(2, 2+n))
        constr.append({'type': 'ineq', 'fun': lambda x: 1 - sum(abs(x[i]) for i in ar_idx)})
        
        pad = max(n, m, p, q)
        y_pad = np.concatenate([np.zeros(pad), self.y])
        
        y_mean = np.mean(self.y)
        y_var = np.var(self.y)
        
        x0 = np.concatenate([
        [y_mean],
        [y_var * 0.1],
        np.zeros(n),
        np.zeros(m),
        np.full(q, 0.1),
        np.full(p, 0.8)
        ])
        
        persistence = sum(x0[i] for i in alpha_idx) + sum(x0[i] for i in beta_idx)
        print(f"Persistence: {persistence}")
        
        res = scipy.optimize.minimize(
            fun=self._get_loglikelihood,
            x0=x0,
            args=(y_pad, y_mean, y_var, pad),
            bounds=bnds,
            constraints=constr,
            method='SLSQP'
        )
        
        for i, (param, bound) in enumerate(zip(res.x, bnds)):
            print(f"param[{i}]: value={param:.4f}, bound={bound}")
    
        if not res.success:
            print("Optimization failed")
        
        self.time = dt.now()
        
        self.params_est = res.x
        self._log_likelihood = -res.fun
        print(f"Log_likelihood: {self._log_likelihood}")
        cov = self._get_cov(lambda params: self._get_loglikelihood(
            params=params,
            y_pad=y_pad,
            y_mean=y_mean,
            y_var=y_var,
            pad=pad
        ))
        
        
        return ModelResults(self, cov)
    




class ModelResults:
    def __init__(self, model: ARMAGARCH, cov_matrix: ndarray):
        self.model = model
        self._cov = cov_matrix

    @property
    def _coef_names(self) -> list[str]:
        n, m, p, q = self.model.order
        names = ['\u03bc', '\u03c9']
        names += [f'\u03c6{i + 1}' for i in range(n)]
        names += [f'\u03b8{i + 1}' for i in range(m)]
        names += [f'\u03b1{i + 1}' for i in range(q)]
        names += [f'\u03b2{i + 1}' for i in range(p)]
        
        return names
    
    @cached_property
    def params(self):
        return self.model.params_est
    
    @cached_property
    def log_likelihood(self) -> float:
        return self.model._log_likelihood
    
    @cached_property
    def std_errors(self) -> ndarray:
        return np.sqrt(np.diag(self._cov))
    
    @cached_property
    def zvalues(self):
        return self.params / self.std_errors
    
    @cached_property
    def pvalues(self):
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(self.zvalues)))
    
    @cached_property
    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([
            self.params - z * self.std_errors,
            self.params + z * self.std_errors
        ])
    
    @cached_property
    def aic(self) -> float:
        num_params = self.model.num_params
        aic = 2 * num_params - 2 * self.log_likelihood
        
        return aic
    
    @cached_property
    def bic(self) -> float:
        num_params = self.model.num_params
        bic = num_params * np.log(len(self.model.y)) - 2 * self.log_likelihood  #y or y_pad??
        
        return bic
    
    def summary(self, alpha = 0.05) -> Summary:
        top_left = [
            ("Dep. Variable:", self.model._dep_var_name),
            ("Model:", f"ARMA-GARCH{self.model.order}"), #TODO: Replace with self.model.name
            ("Date", dt.strftime(self.model.time, "%a, %-d %b %Y")),
            ("Time:", dt.strftime(self.model.time, "%H:%M:%S"))
        ]
        
        top_right = [
            ("No. Observations:", len(self.model.y)),
            ("Log Likelihood:", f"{format_float(self.log_likelihood)}"),
            ("AIC", f"{format_float(self.aic)}"),
            ("BIC", f"{format_float(self.bic)}")
        ]
        
        
        stubs = []
        vals = []
        for stub, val in top_left:
            stubs.append(stub)
            vals.append([val])
        
        table = SimpleTable(data=vals, txt_fmt=fmt_2cols, stubs=stubs, title="ARMA-GARCH Results")
        
        smry = Summary()
        
        fmt = fmt_2cols
        fmt['data_fmts'][1] = '%18s'
        
        top_right = [("%-21s" % ("  " + k), v) for k, v in top_right]
        stubs = []
        vals = []
        for stub, val in top_right:
            stubs.append(stub)
            vals.append([val])
        
        table.extend_right(SimpleTable(data=vals, stubs=stubs))
        smry.tables.append(table)
        
        headers = ["coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]
        stubs = self._coef_names
        vals=[]
        
        for i in range(self.model.num_params):
            vals.append([
                format_float(self.params[i]),
                format_float(self.std_errors[i]),
                format_float(self.zvalues[i]),
                format_float(self.pvalues[i]),
                format_float(self.conf_int[i, 0]),
                format_float(self.conf_int[i, 1])
            ])
        
        
        fmt = fmt_params
        fmt['data_fmts'][1] = '%14s'
            
        smry.tables.append(SimpleTable(data=vals, stubs=stubs, txt_fmt=fmt, headers=headers))
        
        return smry
        



n = 1
m = 0
p = 1
q = 1

true_params = [0.1, 0.1, 0.3, 0.1, 0.8]

y = simulate(1000, order=(n,m,p,q), params=true_params)
# pad = max(n, m, p, q)
# y_pad = np.concatenate([np.zeros(pad), y])

#print(loglikelihood(x0, y_pad, n, m, p, q, y_mean=np.mean(y), y_var=np.var(y), pad=pad))

model = ARMAGARCH(y, order=(n,m,p,q))

res = model.fit()
print(res.summary())

# import arch

# model = arch.univariate.ARX(y, lags=n)
# model.volatility = arch.univariate.GARCH(p,0,q)
# model.distribution = arch.univariate.Normal()
