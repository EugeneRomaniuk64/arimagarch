#include <math.h>
#include <stdbool.h>

static const double LOG_2PI = 1.8378770664093454836;


static inline double fast_log(double x) {
    union { double d; unsigned long long i; } u = { x }; // we can get the raw bits of the double as an integer

    // Extract exponent
    int exp = (int)((u.i >> 52) & 0x7FF) - 1023;
    
    // Force exponent to 0, so mantissa is in [1, 2)
    u.i = (u.i & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m = u.d;
    
    // Minimax polynomial for log(m) on [1, 2), 1 iteration
    double y = (m - 1.0) / (m + 1.0);
    double y2 = y * y;
    double p = y * (2.0 + y2 * (0.66666666666 + y2 * 0.40000000000));
    
    // Recombine: log(x) = log(2^exp * m) = exp*log(2) + log(m)
    return exp * 0.6931471805599453 + p;
}


#define DEFINE_LL(NAME, N, M, P, Q, LOGFN) \
static double NAME( \
    double * restrict data, \
    int data_len, \
    int pad, \
    double mu, \
    double omega, \
    double * restrict phis, \
    double * restrict thetas, \
    double * restrict alphas, \
    double * restrict betas, \
    double * restrict epsilon, \
    double * restrict sigma2 \
) { \
    double ll = 0.0; \
    for (int i = pad; i < data_len; i++) { \
        /*AR terms*/ \
        epsilon[i] = data[i] - mu; \
        for (int j = 0; j < N; j++) \
            epsilon[i] -= phis[j] * data[i - j - 1]; \
        /*MA terms*/ \
        for (int j = 0; j < M; j++) \
            epsilon[i] -= thetas[j] * epsilon[i - j - 1]; \
        /*GARCH terms*/ \
        sigma2[i] = omega; \
        for (int j = 0; j < P; j++) \
            sigma2[i] += betas[j] * sigma2[i - j - 1]; \
        /*ARCH terms*/ \
        for (int j = 0; j < Q; j++) \
            sigma2[i] += alphas[j] * epsilon[i - j - 1] * epsilon[i - j - 1]; \
        if (sigma2[i] <= 0.0) return 1e15; \
        ll -= 0.5 * (LOG_2PI + LOGFN(sigma2[i]) + epsilon[i] * epsilon[i] / sigma2[i]); \
    } \
    return ll; \
} 


// Slow implementations
DEFINE_LL(ll_0000_slow, 0,0,0,0, log)
DEFINE_LL(ll_1000_slow, 1,0,0,0, log)
DEFINE_LL(ll_0100_slow, 0,1,0,0, log)
DEFINE_LL(ll_0010_slow, 0,0,1,0, log)
DEFINE_LL(ll_0001_slow, 0,0,0,1, log)
DEFINE_LL(ll_1100_slow, 1,1,0,0, log)
DEFINE_LL(ll_1010_slow, 1,0,1,0, log)
DEFINE_LL(ll_1001_slow, 1,0,0,1, log)
DEFINE_LL(ll_0101_slow, 0,1,0,1, log)
DEFINE_LL(ll_0110_slow, 0,1,1,0, log)
DEFINE_LL(ll_0011_slow, 0,0,1,1, log)
DEFINE_LL(ll_1110_slow, 1,1,1,0, log)
DEFINE_LL(ll_0111_slow, 0,1,1,1, log)
DEFINE_LL(ll_1011_slow, 1,0,1,1, log)
DEFINE_LL(ll_1101_slow, 1,1,0,1, log)
DEFINE_LL(ll_1111_slow, 1,1,1,1, log)

static double ll_generic_slow(
    double * restrict data, 
    int data_len,
    int pad,
    int n, int m, int p, int q,
    double mu, 
    double omega, 
    double * restrict phis, 
    double * restrict thetas, 
    double * restrict alphas, 
    double * restrict betas, 
    double * restrict epsilon, 
    double * restrict sigma2
) {
    double ll = 0.0; 
    for (int i = pad; i < data_len; i++) { 
        /*AR terms*/ 
        epsilon[i] = data[i] - mu; 
        for (int j = 0; j < n; j++) 
            epsilon[i] -= phis[j] * data[i - j - 1]; 

        /*MA terms*/ 
        for (int j = 0; j < m; j++) 
            epsilon[i] -= thetas[j] * epsilon[i - j - 1]; 

        /*GARCH terms*/ 
        sigma2[i] = omega; 
        for (int j = 0; j < p; j++) 
            sigma2[i] += betas[j] * sigma2[i - j - 1]; 
        
        /*ARCH terms*/ 
        for (int j = 0; j < q; j++) 
            sigma2[i] += alphas[j] * epsilon[i - j - 1] * epsilon[i - j - 1]; 

        if (sigma2[i] <= 0.0) return 1e15; 

        ll -= 0.5 * (LOG_2PI + log(sigma2[i]) + epsilon[i] * epsilon[i] / sigma2[i]); 
    } 
    return ll; 
}


// Fast implementations
DEFINE_LL(ll_0000_fast, 0,0,0,0, fast_log)
DEFINE_LL(ll_1000_fast, 1,0,0,0, fast_log)
DEFINE_LL(ll_0100_fast, 0,1,0,0, fast_log)
DEFINE_LL(ll_0010_fast, 0,0,1,0, fast_log)
DEFINE_LL(ll_0001_fast, 0,0,0,1, fast_log)
DEFINE_LL(ll_1100_fast, 1,1,0,0, fast_log)
DEFINE_LL(ll_1010_fast, 1,0,1,0, fast_log)
DEFINE_LL(ll_1001_fast, 1,0,0,1, fast_log)
DEFINE_LL(ll_0101_fast, 0,1,0,1, fast_log)
DEFINE_LL(ll_0110_fast, 0,1,1,0, fast_log)
DEFINE_LL(ll_0011_fast, 0,0,1,1, fast_log)
DEFINE_LL(ll_1110_fast, 1,1,1,0, fast_log)
DEFINE_LL(ll_0111_fast, 0,1,1,1, fast_log)
DEFINE_LL(ll_1011_fast, 1,0,1,1, fast_log)
DEFINE_LL(ll_1101_fast, 1,1,0,1, fast_log)
DEFINE_LL(ll_1111_fast, 1,1,1,1, fast_log)

static double ll_generic_fast(
    double * restrict data, 
    int data_len,
    int pad,
    int n, int m, int p, int q,
    double mu, 
    double omega, 
    double * restrict phis, 
    double * restrict thetas, 
    double * restrict alphas, 
    double * restrict betas, 
    double * restrict epsilon, 
    double * restrict sigma2 
) {
    double ll = 0.0; 
    for (int i = pad; i < data_len; i++) { 
        /*AR terms*/ 
        epsilon[i] = data[i] - mu; 
        for (int j = 0; j < n; j++) 
            epsilon[i] -= phis[j] * data[i - j - 1]; 

        /*MA terms*/ 
        for (int j = 0; j < m; j++) 
            epsilon[i] -= thetas[j] * epsilon[i - j - 1]; 
        
        /*GARCH terms*/ 
        sigma2[i] = omega; 
        for (int j = 0; j < p; j++) 
            sigma2[i] += betas[j] * sigma2[i - j - 1]; 

        /*ARCH terms*/ 
        for (int j = 0; j < q; j++) 
            sigma2[i] += alphas[j] * epsilon[i - j - 1] * epsilon[i - j - 1]; 

        if (sigma2[i] <= 0.0) return 1e15; 
        
        ll -= 0.5 * (LOG_2PI + fast_log(sigma2[i]) + epsilon[i] * epsilon[i] / sigma2[i]); 
    } 
    return ll; 
}


double log_likelihood (
    double * restrict data,
    int data_len,
    int pad,
    double * restrict params,
    int * restrict order,
    double * restrict epsilon,
    double * restrict sigma2,
    bool optimized
) {
    int n = order[0];
    int m = order[1];
    int p = order[2];
    int q = order[3];

    double mu = params[0];
    double omega = params[1];

    double *phis = params + 2;
    double *thetas = params + 2 + n;
    double *alphas = params + 2 + n + m;
    double *betas = params + 2 + n + m + q;

    int key = n * 1000 + m * 100 + p * 10 + q;
    if (optimized) {
        switch (key)
        {
            case 0:
                return ll_0000_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1000:
                return ll_1000_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 100:
                return ll_0100_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 10:
                return ll_0010_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1:
                return ll_0001_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1100:
                return ll_1100_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 110:
                return ll_0110_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 11:
                return ll_0011_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1010:
                return ll_1010_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 101:
                return ll_0101_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1001:
                return ll_1001_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1110:
                return ll_1110_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 111:
                return ll_0111_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1011:
                return ll_1011_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1101:
                return ll_1101_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1111:
                return ll_1111_fast(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            default:
                return ll_generic_fast(data, data_len, pad, n, m, p, q, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
        }
    } else
    {
        switch (key)
        {
            case 0:
                return ll_0000_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1000:
                return ll_1000_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 100:
                return ll_0100_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 10:
                return ll_0010_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1:
                return ll_0001_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1100:
                return ll_1100_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 110:
                return ll_0110_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 11:
                return ll_0011_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1010:
                return ll_1010_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 101:
                return ll_0101_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1001:
                return ll_1001_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1110:
                return ll_1110_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 111:
                return ll_0111_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1011:
                return ll_1011_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1101:
                return ll_1101_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            case 1111:
                return ll_1111_slow(data, data_len, pad, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
            default:
                return ll_generic_slow(data, data_len, pad, n, m, p, q, mu, omega, phis, thetas, alphas, betas, epsilon, sigma2);
        }
    }
    
    
}