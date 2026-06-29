#ifndef ARMAGARCH_H
#define ARMAGARCH_H

#include <stdbool.h>

double log_likelihood(
    double * restrict data,
    int data_len,
    int pad,
    int * restrict order,
    double * restrict params,
    double * restrict epsilon,
    double * restrict sigma2,
    bool optimized
);

void compute_gradient(
    double * restrict data,
    int data_len,
    int pad,
    int * restrict order,
    double * restrict params,
    double * restrict epsilon,
    double * restrict sigma2,
    double * restrict d_eps,
    double * restrict d_sig2,
    double * restrict grad
);

#endif