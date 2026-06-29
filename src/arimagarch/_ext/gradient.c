
void compute_gradient(
    double * restrict data,
    int data_len,
    int pad,
    int * restrict order,
    double * restrict params,
    double * restrict epsilon, 
    double * restrict sigma2,
    double * restrict d_eps, // [data_len, n_params]
    double * restrict d_sig2, // [data_len, n_params]
    double * restrict grad // [n_params]
){ 
    int n = order[0];
    int m = order[1];
    int p = order[2];
    int q = order[3];

    int n_params = 2 + n + m + p + q;

    double mu = params[0];
    double omega = params[1];

    double *phis = params + 2;
    double *thetas = params + 2 + n;
    double *alphas = params + 2 + n + m;
    double *betas = params + 2 + n + m + q;

    for (int t = pad; t < data_len; t++)
    {
        // Derivative of epsilon w.r.t. mu
        d_eps[t * n_params] = -1;
        for (int j = 0; j < m; j++)
        {
            d_eps[t * n_params] -= thetas[j] * d_eps[(t - j - 1) * n_params];
        }

        // Derivative of sigma^2 w.r.t. mu
        for (int j = 0; j < q; j++) 
        {
            d_sig2[t * n_params] += 2 * alphas[j] * epsilon[t - j -1] * d_eps[(t - j - 1) * n_params];
        }
        
        for (int j = 0; j < p; j++)
        {
            d_sig2[t * n_params] += betas[j] * d_sig2[(t - j - 1) * n_params];
        }
        

        // Derivative of sigma^2 w.r.t. omega
        d_sig2[t * n_params + 1] = 1;
        for (int j = 0; j < p; j++)
        {
            d_sig2[t * n_params + 1] += betas[j] * d_sig2[(t - j - 1) * n_params + 1];
        }
        

        for (int i = 0; i < n; i++)
        {
            // Derivative of epsilon w.r.t. phi_i
            d_eps[t * n_params + 2 + i] = -data[t - i - 1];
            for (int j = 0; j < m; j++)
            {
                d_eps[t * n_params + 2 + i] -= thetas[j] * d_eps[(t - j - 1) * n_params + 2 + i];
            }
            // Derivative of sigma^2 w.r.t. phi_i
            for (int j = 0; j < q; j++)
            {
                d_sig2[t * n_params + 2 + i] += 2 * alphas[j] * epsilon[t - j - 1] * d_eps[(t - j - 1) * n_params + 2 + i];
            }
            for (int j = 0; j < p; j++)
            {
                d_sig2[t * n_params + 2 + i] += betas[j] * d_sig2[(t - j - 1) * n_params + 2 + i];
            }
        }


        for (int i = 0; i < m; i++)
        {
            // Derivative of epsilon w.r.t. theta_i
            d_eps[t * n_params + 2 + n + i] = -epsilon[t - i - 1];
            for (int j = 0; j < m; j++)
            {
                d_eps[t * n_params + 2 + n + i] -= thetas[j] * d_eps[(t - j - 1) * n_params + 2 + n + i];
            }
            // Derivative of sigma^2 w.r.t. theta_i
            for (int j = 0; j < q; j++)
            {
                d_sig2[t * n_params + 2 + n + i] += 2 * alphas[j] * epsilon[t - j - 1] * d_eps[(t - j - 1) * n_params + 2 + n + i];
            }
            for (int j = 0; j < p; j++)
            {
                d_sig2[t * n_params + 2 + n + i] += betas[j] * d_sig2[(t - j - 1) * n_params + 2 + n + i];
            }
        }



        for (int i = 0; i < q; i++)
        {
            // Derivative of epsilon w.r.t. alpha_i
            d_eps[t * n_params + 2 + n + m + i] = 0;
            
            // Derivative of sigma^2 w.r.t. alpha_i
            d_sig2[t * n_params + 2 + n + m + i] = epsilon[t - i - 1] * epsilon[t - i - 1];
            for (int j = 0; j < p; j++)
            {
                d_sig2[t * n_params + 2 + n + m + i] += betas[j] * d_sig2[(t - j - 1) * n_params + 2 + n + m + i];
            }
        }

        for (int i = 0; i < p; i++)
        {
            // Derivative of epsilon w.r.t. beta_i
            d_eps[t * n_params + 2 + n + m + q + i] = 0;
            
            // Derivative of sigma^2 w.r.t. beta_i
            d_sig2[t * n_params + 2 + n + m + q + i] = sigma2[t - i - 1];
            for (int j = 0; j < p; j++)
            {
                d_sig2[t * n_params + 2 + n + m + q + i] += betas[j] * d_sig2[(t - j - 1) * n_params + 2 + n + m + q + i];
            }
        }
        

        for (int i = 0; i < n_params; i++)
        {
            grad[i] += -d_sig2[t * n_params + i] / (2 * sigma2[t])
                    - (epsilon[t] * d_eps[t * n_params + i]) / sigma2[t]
                    + (epsilon[t] * epsilon[t] * d_sig2[t * n_params + i]) / (2 * sigma2[t] * sigma2[t]);
        }
    }
}
