#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "arimagarch.h"


static PyObject* py_log_likelihood(PyObject *self, PyObject *args)
{
    PyArrayObject *data, *order, *params, *epsilon, *sigma2;
    int data_len, pad, optimized;

    if (!PyArg_ParseTuple(args, "O!iiO!O!O!O!p", 
        &PyArray_Type, &data,
        &data_len,
        &pad,
        &PyArray_Type, &order,
        &PyArray_Type, &params,
        &PyArray_Type, &epsilon,
        &PyArray_Type, &sigma2,
        &optimized
    ))
    {
        return NULL;
    }
    
    double ll = log_likelihood(
        (double*)PyArray_DATA(data),
        data_len,
        pad,
        (int*)PyArray_DATA(order),
        (double*)PyArray_DATA(params),
        (double*)PyArray_DATA(epsilon),
        (double*)PyArray_DATA(sigma2),
        optimized
    );

    return PyFloat_FromDouble(ll);
}

static PyObject* py_compute_gradient(PyObject *self, PyObject *args)
{
    PyArrayObject *data, *order, *params, *epsilon, *sigma2, *d_eps, *d_sig2, *grad;
    int data_len, pad;
    
    if (!PyArg_ParseTuple(args, "O!iiO!O!O!O!O!O!O!", 
        &PyArray_Type, &data,
        &data_len,
        &pad,
        &PyArray_Type, &order,
        &PyArray_Type, &params,
        &PyArray_Type, &epsilon,
        &PyArray_Type, &sigma2,
        &PyArray_Type, &d_eps,
        &PyArray_Type, &d_sig2,
        &PyArray_Type, &grad
    ))
    {
        return NULL;
    }

    compute_gradient(
        (double*)PyArray_DATA(data),
        data_len,
        pad,
        (int*)PyArray_DATA(order),
        (double*)PyArray_DATA(params),
        (double*)PyArray_DATA(epsilon),
        (double*)PyArray_DATA(sigma2),
        (double*)PyArray_DATA(d_eps),
        (double*)PyArray_DATA(d_sig2),
        (double*)PyArray_DATA(grad)
    );

    Py_RETURN_NONE;
}


static PyMethodDef ArimaGarchMethods[] = {
    {
        "log_likelihood",
        py_log_likelihood,
        METH_VARARGS,
        "Compute log-likelihood"
    },
    {
        "compute_gradient",
        py_compute_gradient,
        METH_VARARGS,
        "Compute analytical gradient"
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef arimagarchmodule = {
    PyModuleDef_HEAD_INIT,
    "_arimagarch",
    NULL,
    -1,
    ArimaGarchMethods
};

PyMODINIT_FUNC PyInit__arimagarch(void) {
    import_array();
    return PyModule_Create(&arimagarchmodule);
}