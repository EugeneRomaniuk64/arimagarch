from __future__ import annotations

import numpy as np
import numdifftools as nd
import scipy.optimize
import warnings
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary, fmt_2cols, fmt_params
from datetime import datetime as dt
from functools import cached_property
from numpy.typing import NDArray
from typing import Literal


from arimagarch import _arimagarch


def format_float(number: float) -> str:
    return f"{number:.3f}"

class ARMAGARCH:
    def __init__(self, data, order: tuple[int]):
        self.data = data
        self.order = np.asarray(order, dtype=np.int32)
        self.n_params = sum(order) + 2 # add 2 for mu and omega
        self.data_len = len(data)
        
        self._data_mean = np.mean(data)
        self._data_var = np.var(data)
        
        self._dep_var_name = None #TODO: Add var name for pandas
        
        self._pad = max(order)
        self._data_pad = np.concatenate([np.zeros(self._pad), self.data])
        self._data_pad_len = len(self._data_pad)
        
        # Initializing buffers
        self._epsilon = np.zeros(self._data_pad_len, dtype=np.float64)
        self._sigma2 = np.concatenate([
            np.full(self._pad, self._data_var),
            np.zeros(self.data_len)
        ]).astype(np.float64)
        self._sigma2_init = self._sigma2.copy() # Needed to reset sigma2 fast
        self._d_eps = np.zeros((self._data_pad_len, self.n_params), dtype=np.float64)
        self._d_sig2 = np.zeros((self._data_pad_len, self.n_params), dtype=np.float64)
        self._grad = np.zeros(self.n_params, dtype=np.float64)
        
        
    def _get_loglikelihood(
        self,
        params: tuple[float], 
        optimized: bool,
        grad_method: str
    ) -> float | tuple[float, NDArray]:
        """Log-likelood computation for ARMA(n,m)-GARCH(p,q) model

        Args:
            params (tuple[float]): model parameters
            optimized (bool): true for fast log computation
            grad_method (str): analytic vs numerical gradient

        Returns:
            float: negative log-likehood for the data
            tuple[float, ndarray]: negative log-likelihood and analytic gradient
        """
        params = np.asarray(params, dtype=np.float64)
        
        # Reset the buffers
        self._epsilon[:] = 0
        np.copyto(self._sigma2, self._sigma2_init)
        self._d_eps[:] = 0
        self._d_sig2[:] = 0
        
        ll = _arimagarch.log_likelihood(
            self._data_pad,
            self._data_pad_len,
            self._pad,
            self.order,
            params,
            self._epsilon,
            self._sigma2,
            optimized
        )
        
        if grad_method == 'analytic':
            self._grad[:] = 0
            
            if -ll >= 1e15:
                return np.float64(-ll), -self._grad.copy()
            
            _arimagarch.compute_gradient(
                self._data_pad,
                self._data_pad_len,
                self._pad,
                self.order,
                params,
                self._epsilon,
                self._sigma2,
                self._d_eps,
                self._d_sig2,
                self._grad
            )
            
            return np.float64(-ll), -self._grad.copy()
        
        else:
            return np.float64(-ll)
    
        
    def _get_cov(self, use_analytic_grad: bool) -> NDArray:
        """Computes the covariance matrix for the log likehood function

        Raises:
            ValueError: Variance cannon be negative

        Returns:
            np.ndarray: an array representing the covariance matrix
        """
        def ll_fixed(params):
            ll = self._get_loglikelihood(params, optimized=False, grad_method='numerical')
            return ll
        
        def grad_fixed(params):
            _, g = self._get_loglikelihood(params, optimized=False, grad_method='analytic')
            return g
        
        try:
            if use_analytic_grad:
                H = nd.Jacobian(grad_fixed)(self.params_est)
            else:
                H = nd.Hessian(ll_fixed)(self.params_est)
                
            cov = np.linalg.pinv(H)
            
            if np.any(np.diag(cov) < 0):
                raise ValueError("Non-positive variance estimates")
            
            return cov
        except Exception as e:
            warnings.warn(f"Could not compute covariance matrix: {e}")
            return np.full((len(self.params_est), len(self.params_est)), np.nan)

    
    def fit(self, optimized: bool = True, grad_method: Literal['analytic', 'numerical'] = 'analytic') -> ModelResults:
        """Fits an ARMA(n,m)-GARCH(p,q) model
        
        Args:
            optimized (bool, optional): true to use fast log algorith (has a larger error). Defaults to True.
            grad_method (Literal[analytic, numerical], optional): method of gradient computation for optimization. Defaults to 'analytic'.

        Returns:
            ModelResults: object containing the results of the fit
        """
        
        use_analytic_grad = True if grad_method == 'analytic' else False
        
        n, m, p, q = self.order
        
        bnds = (
            [(None, None)] +
            [(1e-6, None)] +
            [(None, None)] * n +
            [(None, None)] * m +
            [(1e-6, 0.999)] * p +
            [(1e-6, 0.999)] * q
        )
            

        x0 = np.concatenate([
            [self._data_mean],
            [self._data_var * 0.1],
            np.zeros(n),
            np.zeros(m),
            np.full(q, 0.05),
            np.full(p, 0.5)
        ])
        
        

        res = scipy.optimize.minimize(
            fun=self._get_loglikelihood,
            x0=x0,
            args=(optimized, grad_method),
            bounds=bnds,
            method='L-BFGS-B',
            jac=use_analytic_grad,
            options={
                'ftol': 1e-12,
                'gtol': 1e-7,
                'maxiter': 1000
            }
        )
     
        
        if not res.success:
            print("Optimization failed")
   
        
        self.params_est = res.x
        self._log_likelihood = -res.fun

        cov = self._get_cov(use_analytic_grad)
        
        self.time = dt.now()
        
        return ModelResults(self, cov)
    


class ModelResults:
    def __init__(self, model: ARMAGARCH, cov_matrix: NDArray):
        self.model = model
        self._cov = cov_matrix

    @cached_property
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
    def std_errors(self) -> NDArray:
        return np.sqrt(np.diag(self._cov))
    
    @cached_property
    def zvalues(self):
        return self.params / self.std_errors
    
    @cached_property
    def pvalues(self):
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(self.zvalues)))

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([
            self.params - z * self.std_errors,
            self.params + z * self.std_errors
        ])
    
    @cached_property
    def aic(self) -> float:
        return 2 * self.model.n_params - 2 * self.log_likelihood
    
    @cached_property
    def bic(self) -> float:
        return self.model.n_params * np.log(self.model.data_len) - 2 * self.log_likelihood 
    
    def summary(self, alpha = 0.05) -> Summary:
        top_left = [
            ("Dep. Variable:", self.model._dep_var_name),
            ("Model:", f"ARMA-GARCH{self.model.order}"), #TODO: Replace with self.model.name
            ("Date", dt.strftime(self.model.time, "%a, %-d %b %Y")),
            ("Time:", dt.strftime(self.model.time, "%H:%M:%S"))
        ]
        
        top_right = [
            ("No. Observations:", self.model.data_len),
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
        
        for i in range(self.model.n_params):
            vals.append([
                format_float(self.params[i]),
                format_float(self.std_errors[i]),
                format_float(self.zvalues[i]),
                format_float(self.pvalues[i]),
                format_float(self.conf_int(alpha)[i, 0]),
                format_float(self.conf_int(alpha)[i, 1])
            ])
        
        
        fmt = fmt_params
        fmt['data_fmts'][1] = '%14s'
            
        smry.tables.append(SimpleTable(data=vals, stubs=stubs, txt_fmt=fmt, headers=headers))
        
        return smry
        