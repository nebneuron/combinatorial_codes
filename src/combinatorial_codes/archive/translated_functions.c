// src/translated_functions.c
// Build with: python setup.py build_ext --inplace
// Exposes:
//   generate_increasing_tuples(m, k)        -> ndarray[intp]
//   custom_bit_length(x)                    -> int
//   x_is_a_superset_of_any_in_list(x, L)    -> bool
//   lattice_slice2(m, k, minimal_non_faces) -> ndarray[uint64]

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>

/*----------------------------------------------------------------------
 * Helpers
 *----------------------------------------------------------------------*/
static int64_t binom_int64(int64_t n, int64_t k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    int64_t c = 1;
    for (int64_t i = 1; i <= k; ++i)
        c = (c * (n - (k - i))) / i;
    return c;
}

/* Return 1 if (b & ~x) == 0 for any b in seq, 0 otherwise, -1 on error */
static int is_superset(unsigned long long x, PyObject *seq_fast) {
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq_fast);
    PyObject **items = PySequence_Fast_ITEMS(seq_fast);
    for (Py_ssize_t i = 0; i < n; ++i) {
        unsigned long long b = PyLong_AsUnsignedLongLong(items[i]);
        if (PyErr_Occurred()) return -1;
        if ((b & ~x) == 0ULL) return 1;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/*  intersection_of_codewords_from_bits(x, a)                                 */
/* -------------------------------------------------------------------------- */
static uint64_t intersect_mask_with_words(unsigned long long mask,
                                          const uint64_t *words,
                                          npy_uintp len,
                                          int *err)
{
    if (mask == 0ULL) {
        *err = 1; return 0ULL;
    }
    uint64_t res = UINT64_MAX; int init = 0; npy_uintp idx = 0;
    unsigned long long tmp = mask;
    while (tmp) {
        if (tmp & 1ULL) {
            if (idx >= len) { *err = 1; return 0ULL; }
            res = init ? (res & words[idx]) : words[idx];
            init = 1;
            if (res == 0ULL) return 0ULL; /* early exit */
        }
        tmp >>= 1ULL; ++idx;
    }
    *err = 0; return res;
}



/*----------------------------------------------------------------------
 * generate_increasing_tuples
 *----------------------------------------------------------------------*/
static PyObject *py_generate_increasing_tuples(PyObject *self, PyObject *args) {
    long long m_in, k_in;
    if (!PyArg_ParseTuple(args, "LL", &m_in, &k_in)) return NULL;
    if (m_in < 0 || k_in < 0 || k_in > m_in) {
        PyErr_SetString(PyExc_ValueError, "Require 0 <= k <= m and m,k >= 0");
        return NULL;
    }
    int64_t m = (int64_t)m_in, k = (int64_t)k_in;
    int64_t ncomb = binom_int64(m, k);
    npy_intp dims[2] = {(npy_intp)ncomb, (npy_intp)k};
    PyObject *arr = PyArray_SimpleNew(2, dims, NPY_INTP);
    if (!arr) return NULL;
    npy_intp *data = (npy_intp *)PyArray_DATA((PyArrayObject *)arr);

    if (k > 64) {
        Py_DECREF(arr);
        PyErr_SetString(PyExc_ValueError, "k too large; adjust buffer size");
        return NULL;
    }
    npy_intp current[64];
    for (npy_intp i = 0; i < k; ++i) current[i] = i;
    npy_intp row = 0;
    int done = 0;
    while (!done) {
        for (npy_intp j = 0; j < k; ++j) data[row * k + j] = current[j];
        ++row;
        npy_intp idx = k - 1;
        while (idx >= 0 && current[idx] == m - k + idx) --idx;
        if (idx < 0) done = 1;
        else {
            ++current[idx];
            for (npy_intp j = idx + 1; j < k; ++j) current[j] = current[j - 1] + 1;
        }
    }
    return arr;
}

/*----------------------------------------------------------------------
 * custom_bit_length
 *----------------------------------------------------------------------*/
static PyObject *py_custom_bit_length(PyObject *self, PyObject *args) {
    unsigned long long x;
    if (!PyArg_ParseTuple(args, "K", &x)) return NULL;
    if (x == 0) return PyLong_FromLong(0);
    int len = 0;
    while (x) { ++len; x >>= 1; }
    return PyLong_FromLong(len);
}

/*----------------------------------------------------------------------
 * x_is_a_superset_of_any_in_list
 *----------------------------------------------------------------------*/
static PyObject *py_x_is_superset_of_any(PyObject *self, PyObject *args) {
    PyObject *x_obj, *seq_obj;
    if (!PyArg_ParseTuple(args, "OO", &x_obj, &seq_obj)) return NULL;
    unsigned long long x = PyLong_AsUnsignedLongLong(x_obj);
    if (PyErr_Occurred()) return NULL;
    PyObject *seq_fast = PySequence_Fast(seq_obj, "L must be a sequence");
    if (!seq_fast) return NULL;
    int res = is_superset(x, seq_fast);
    Py_DECREF(seq_fast);
    if (res < 0) return NULL;
    return PyBool_FromLong(res);
}

/*----------------------------------------------------------------------
 * lattice_slice2
 *----------------------------------------------------------------------*/
static PyObject *py_lattice_slice2(PyObject *self, PyObject *args) {
    long long m_in, k_in; PyObject *minfaces_obj;
    if (!PyArg_ParseTuple(args, "LLO", &m_in, &k_in, &minfaces_obj)) return NULL;
    long long m = m_in, k = k_in;
    if (k <= 1) { PyErr_Format(PyExc_ValueError, "The size k=%lld is not larger than 1", k); return NULL; }
    if (k > m)  { PyErr_Format(PyExc_ValueError, "The size k=%lld is larger than m=%lld", k, m); return NULL; }
    if (m > 64) { PyErr_SetString(PyExc_ValueError, "m > 64 not supported"); return NULL; }

    PyObject *minfaces_fast = PySequence_Fast(minfaces_obj, "minimal_non_faces must be a sequence");
    if (!minfaces_fast) return NULL;

    /* collect uint64 results first */
    PyObject *tmp_list = PyList_New(0);
    if (!tmp_list) { Py_DECREF(minfaces_fast); return NULL; }

    unsigned char current[64];
    for (long long i = 0; i < k; ++i) current[i] = (unsigned char)i;
    int done = 0;
    while (!done) {
        unsigned long long x = 0ULL;
        for (long long j = 0; j < k; ++j) x |= (1ULL << current[j]);
        int super = 0;
        if (PySequence_Fast_GET_SIZE(minfaces_fast) > 0) {
            super = is_superset(x, minfaces_fast);
            if (super < 0) { Py_DECREF(minfaces_fast); Py_DECREF(tmp_list); return NULL; }
        }
        if (!super) {
            PyObject *x_py = PyLong_FromUnsignedLongLong(x);
            if (!x_py || PyList_Append(tmp_list, x_py) < 0) { Py_XDECREF(x_py); Py_DECREF(minfaces_fast); Py_DECREF(tmp_list); return NULL; }
            Py_DECREF(x_py);
        }
        long long idx = k - 1;
        while (idx >= 0 && current[idx] == m - k + idx) --idx;
        if (idx < 0) done = 1;
        else { ++current[idx]; for (long long j = idx + 1; j < k; ++j) current[j] = current[j - 1] + 1; }
    }

    /* convert tmp_list -> ndarray[uint64] */
    Py_ssize_t n_out = PyList_GET_SIZE(tmp_list);
    npy_intp dims[1] = {(npy_intp)n_out};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!arr) { Py_DECREF(minfaces_fast); Py_DECREF(tmp_list); return NULL; }
    uint64_t *arr_data = (uint64_t *)PyArray_DATA((PyArrayObject *)arr);
    for (Py_ssize_t i = 0; i < n_out; ++i) {
        PyObject *item = PyList_GET_ITEM(tmp_list, i);
        arr_data[i] = (uint64_t)PyLong_AsUnsignedLongLong(item);
        if (PyErr_Occurred()) { Py_DECREF(minfaces_fast); Py_DECREF(tmp_list); Py_DECREF(arr); return NULL; }
    }

    Py_DECREF(minfaces_fast);
    Py_DECREF(tmp_list);
    return arr;
}


/* ------------------------------------------------------------------------- */
/* intersection_of_codewords_from_bits(x, a)                                  */
/* ------------------------------------------------------------------------- */
static PyObject *py_intersection_of_codewords(PyObject *self, PyObject *args)
{
    unsigned long long x; PyObject *arr_obj;
    if (!PyArg_ParseTuple(args, "KO", &x, &arr_obj)) return NULL;
    if (x == 0ULL) {
        PyErr_SetString(PyExc_ValueError, "x must be non-zero bitmask");
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(arr_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;
    if (PyArray_NDIM(arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "a must be 1-D uint64 array");
        Py_DECREF(arr); return NULL;
    }
    npy_uintp len = PyArray_SIZE(arr);
    const uint64_t *data = (const uint64_t *)PyArray_DATA(arr);

    uint64_t res = UINT64_MAX; int init = 0; npy_uintp idx = 0;
    unsigned long long tmp = x;
    while (tmp) {
        if (tmp & 1ULL) {
            if (idx >= len) {
                PyErr_SetString(PyExc_ValueError, "bit in x exceeds array length");
                Py_DECREF(arr); return NULL;
            }
            res = init ? (res & data[idx]) : data[idx];
            init = 1;
            if (res == 0ULL) { Py_DECREF(arr); return PyLong_FromUnsignedLongLong(0ULL); }
        }
        tmp >>= 1ULL; ++idx;
    }
    Py_DECREF(arr);
    return PyLong_FromUnsignedLongLong(res);
}



/*----------------------------------------------------------------------
 * Module definition
 *----------------------------------------------------------------------*/
static PyMethodDef Methods[] = {
    {"generate_increasing_tuples", py_generate_increasing_tuples, METH_VARARGS, "Generate all increasing k‑tuples"},
    {"custom_bit_length",         py_custom_bit_length,          METH_VARARGS, "Bit length of unsigned int"},
    {"x_is_a_superset_of_any_in_list", py_x_is_superset_of_any,  METH_VARARGS, "Return True if x supersets any b in list"},
    {"lattice_slice2",            py_lattice_slice2,             METH_VARARGS, "Return uint64 ndarray of valid k-sets"},
    {"intersection_of_codewords_from_bits", py_intersection_of_codewords, METH_VARARGS, "AND‑intersect selected codewords"},
    {NULL, NULL, 0, NULL}
};



static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "translated_functions",
    "C translations of performance‑critical Numba functions",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_translated_functions(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    import_array();
    return m;
}
