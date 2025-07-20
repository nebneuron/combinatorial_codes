// translated_functions.c
// Build with: python setup.py build_ext --inplace
// C translations of performance‑critical helpers originally in Numba.
// Now *fully* includes: lattice_slice2, generate_increasing_tuples, custom_bit_length,
// x_is_a_superset_of_any_in_list, intersection_of_codewords_from_bits,
// intersections_inside_a_clique.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>   /* for memcpy used later */

/* forward declaration so qsort sees it */
static int cmp_u64(const void *a, const void *b);
/* ========================================================================== */
/*  Small utilities                                                           */
/* ========================================================================== */

static int64_t binom_int64(int64_t n, int64_t k)
{
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    int64_t c = 1;
    for (int64_t i = 1; i <= k; ++i)
        c = (c * (n - (k - i))) / i;
    return c;
}

/* Convenience for raising */
static inline void raise_simple(const char *msg) { PyErr_SetString(PyExc_ValueError, msg); }

/* Helper: is_superset(mask, list_of_masks) – returns 1/0, -1 on error */
static int is_superset(unsigned long long mask, PyObject *seq_fast)
{
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq_fast);
    PyObject **items = PySequence_Fast_ITEMS(seq_fast);
    for (Py_ssize_t i = 0; i < n; ++i) {
        unsigned long long b = PyLong_AsUnsignedLongLong(items[i]);
        if (PyErr_Occurred()) return -1;
        if ((b & ~mask) == 0ULL) return 1; /* b ⊆ mask */
    }
    return 0;
}

/* ========================================================================== */
/*  generate_increasing_tuples                                                */
/* ========================================================================== */
static PyObject *py_generate_increasing_tuples(PyObject *self, PyObject *args)
{
    long long m_in, k_in;
    if (!PyArg_ParseTuple(args, "LL", &m_in, &k_in)) return NULL;
    if (m_in < 0 || k_in < 0 || k_in > m_in) { raise_simple("Require 0 <= k <= m"); return NULL; }
    int64_t m = (int64_t)m_in, k = (int64_t)k_in;
    int64_t ncomb = binom_int64(m, k);
    npy_intp dims[2] = {(npy_intp)ncomb, (npy_intp)k};
    PyObject *arr = PyArray_SimpleNew(2, dims, NPY_INTP);
    if (!arr) return NULL;
    npy_intp *data = (npy_intp *)PyArray_DATA((PyArrayObject *)arr);

    npy_intp idx[64];
    if (k > 64) { Py_DECREF(arr); raise_simple("k too large"); return NULL; }
    for (npy_intp i = 0; i < k; ++i) idx[i] = i;
    npy_intp row = 0; int done = 0;
    while (!done) {
        for (npy_intp j = 0; j < k; ++j) data[row * k + j] = idx[j];
        ++row;
        npy_intp pos = k - 1;
        while (pos >= 0 && idx[pos] == m - k + pos) --pos;
        if (pos < 0) done = 1;
        else {
            ++idx[pos];
            for (npy_intp j = pos + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
        }
    }
    return arr;
}

/* ========================================================================== */
/*  custom_bit_length                                                         */
/* ========================================================================== */
static PyObject *py_custom_bit_length(PyObject *self, PyObject *args)
{
    unsigned long long x; if (!PyArg_ParseTuple(args, "K", &x)) return NULL;
    int n = 0; while (x) { ++n; x >>= 1ULL; }
    return PyLong_FromLong(n);
}

/* ========================================================================== */
/*  x_is_a_superset_of_any_in_list                                           */
/* ========================================================================== */
static PyObject *py_x_is_superset_of_any(PyObject *self, PyObject *args)
{
    unsigned long long x; PyObject *seq;
    if (!PyArg_ParseTuple(args, "KO", &x, &seq)) return NULL;
    PyObject *fast = PySequence_Fast(seq, "need iterable"); if (!fast) return NULL;
    int out = is_superset(x, fast); Py_DECREF(fast);
    if (out == -1) return NULL;
    return PyBool_FromLong(out);
}

/* ========================================================================== */
/*  intersection_of_codewords_from_bits                                      */
/* ========================================================================== */
static uint64_t intersect_with_words(unsigned long long mask, const uint64_t *words, npy_uintp n, int *err)
{
    if (mask == 0ULL) { *err = 1; return 0ULL; }
    uint64_t res = UINT64_MAX; int init = 0; npy_uintp idx = 0;
    while (mask) {
        if (mask & 1ULL) {
            if (idx >= n) { *err = 1; return 0ULL; }
            res = init ? (res & words[idx]) : words[idx];
            init = 1; if (res == 0ULL) return 0ULL;
        }
        mask >>= 1ULL; ++idx;
    }
    *err = 0; return res;
}

static PyObject *py_intersection_of_codewords(PyObject *self, PyObject *args)
{
    unsigned long long mask; PyObject *arr_obj;
    if (!PyArg_ParseTuple(args, "KO", &mask, &arr_obj)) return NULL;
    PyArrayObject *arr = (PyArrayObject *)PyArray_FROM_OTF(arr_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;
    const uint64_t *data = (const uint64_t *)PyArray_DATA(arr);
    npy_uintp n = PyArray_SIZE(arr);
    int err = 0; uint64_t r = intersect_with_words(mask, data, n, &err);
    Py_DECREF(arr);
    if (err) return NULL;
    return PyLong_FromUnsignedLongLong(r);
}

/* ========================================================================== */
/*  lattice_slice2 (implicit vertex set 0..m-1)                              */
/* ========================================================================== */
static PyObject *py_lattice_slice2(PyObject *self, PyObject *args)
{
    long long m_in, k_in; PyObject *mnf_obj;
    if (!PyArg_ParseTuple(args, "LLO", &m_in, &k_in, &mnf_obj)) return NULL;
    if (m_in < 0 || k_in < 1 || k_in > m_in) { raise_simple("invalid m/k"); return NULL; }
    int64_t m = (int64_t)m_in, k = (int64_t)k_in;

    PyObject *fast = PySequence_Fast(mnf_obj, "need iterable"); if (!fast) return NULL;

    /* Precompute shifts */
    if (m > 64) { Py_DECREF(fast); raise_simple("m>64 not supported"); return NULL; }
    uint64_t shifts[64]; for (int64_t i = 0; i < m; ++i) shifts[i] = UINT64_C(1) << i;

    /* Enumerate combinations and collect */
    PyObject *list = PyList_New(0); if (!list) { Py_DECREF(fast); return NULL; }

    int64_t idx[64]; for (int64_t i = 0; i < k; ++i) idx[i] = i;
    int done = 0;
    while (!done) {
        unsigned long long mask = 0ULL;
        for (int64_t j = 0; j < k; ++j) mask |= shifts[idx[j]];
        int sup = is_superset(mask, fast);
        if (sup == -1) { Py_DECREF(fast); Py_DECREF(list); return NULL; }
        if (!sup) {
            PyObject *py = PyLong_FromUnsignedLongLong(mask);
            if (!py || PyList_Append(list, py) == -1) { Py_XDECREF(py); Py_DECREF(fast); Py_DECREF(list); return NULL; }
            Py_DECREF(py);
        }
        int64_t pos = k - 1;
        while (pos >= 0 && idx[pos] == m - k + pos) --pos;
        if (pos < 0) done = 1;
        else {
            ++idx[pos];
            for (int64_t j = pos + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
        }
    }
    Py_DECREF(fast);

    /* Convert list to numpy uint64 array */
    Py_ssize_t n = PyList_GET_SIZE(list);
    npy_intp dims[1] = {(npy_intp)n};
    PyObject *arr = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!arr) { Py_DECREF(list); return NULL; }
    uint64_t *out = (uint64_t *)PyArray_DATA((PyArrayObject *)arr);
    for (Py_ssize_t i = 0; i < n; ++i) {
        out[i] = PyLong_AsUnsignedLongLong(PyList_GET_ITEM(list, i));
    }
    Py_DECREF(list);
    return arr;
}

/* ========================================================================== */
/*  intersections_inside_a_clique                                            */
/* ========================================================================== */
static int mask_contains_forbidden(unsigned long long mask, PyObject *mnf)
{
    PyObject *fast = PySequence_Fast(mnf, "iter"); if (!fast) return -1;
    int r = is_superset(mask, fast); Py_DECREF(fast); return r;
}

static PyObject *py_intersections_inside_a_clique(PyObject *self, PyObject *args)
{
    PyObject *max_obj, *clique_obj, *mnf_obj;
    if (!PyArg_ParseTuple(args, "OOO", &max_obj, &clique_obj, &mnf_obj)) return NULL;

    PyArrayObject *max_arr = (PyArrayObject *)PyArray_FROM_OTF(max_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *cli_arr = (PyArrayObject *)PyArray_FROM_OTF(clique_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!max_arr || !cli_arr) { Py_XDECREF(max_arr); Py_XDECREF(cli_arr); return NULL; }

    npy_uintp m = PyArray_SIZE(cli_arr);
    if (m > 64) { Py_DECREF(max_arr); Py_DECREF(cli_arr); raise_simple("clique>64"); return NULL; }

    const uint64_t *max_data = (const uint64_t *)PyArray_DATA(max_arr);
    npy_uintp max_len = PyArray_SIZE(max_arr);
    const uint64_t *cli_data = (const uint64_t *)PyArray_DATA(cli_arr);

    uint64_t shifts[64]; for (npy_uintp i = 0; i < m; ++i) shifts[i] = UINT64_C(1) << cli_data[i];

    PyObject *ilist = PyList_New(0); if (!ilist) { Py_DECREF(max_arr); Py_DECREF(cli_arr); return NULL; }

    for (npy_uintp k = 2; k <= m; ++k) {
        npy_intp idxs[64]; for (npy_uintp i = 0; i < k; ++i) idxs[i] = i;
        int done = 0; npy_uintp n_cand = 0, n_rej = 0;
        while (!done) {
            unsigned long long mask = 0ULL;
            for (npy_uintp j = 0; j < k; ++j) mask |= shifts[idxs[j]];
            int sup = mask_contains_forbidden(mask, mnf_obj);
            if (sup == -1) goto fail;
            if (!sup) {
                ++n_cand;
                int err = 0; uint64_t inter = intersect_with_words(mask, max_data, max_len, &err);
                if (err) { raise_simple("mask bit out of range"); goto fail; }
                if (inter) {
                    PyObject *py = PyLong_FromUnsignedLongLong(inter);
                    if (!py || PyList_Append(ilist, py) == -1) { Py_XDECREF(py); goto fail; }
                    Py_DECREF(py);
                } else {
                    if (!PyObject_CallMethod(mnf_obj, "append", "K", mask)) goto fail;
                    ++n_rej;
                }
            }
            /* next comb */
            npy_intp pos = k - 1;
            while (pos >= 0 && idxs[pos] == (npy_intp)(m - k + pos)) --pos;
            if (pos < 0) done = 1;
            else { ++idxs[pos]; for (npy_uintp j = pos + 1; j < k; ++j) idxs[j] = idxs[j - 1] + 1; }
        }
        if (n_cand == 0 || n_rej == n_cand) break;
    }

    /* numpy.unique(ilist) */
    PyObject *np_mod = PyImport_ImportModule("numpy"); if (!np_mod) goto fail;
    PyObject *unique = PyObject_GetAttrString(np_mod, "unique"); Py_DECREF(np_mod);
    if (!unique) goto fail;
    PyObject *ilist_arr = PyArray_FROM_OTF(ilist, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(ilist);
    PyObject *out = PyObject_CallFunctionObjArgs(unique, ilist_arr, NULL);
    Py_DECREF(unique); Py_DECREF(ilist_arr); Py_DECREF(max_arr); Py_DECREF(cli_arr);
    return out;

fail:
    Py_XDECREF(ilist); Py_DECREF(max_arr); Py_DECREF(cli_arr);
    return NULL;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* ========================================================================= */
/* intersections_work                                                         */
/* ========================================================================= */
/* Signature on the Python side:
intersections_work(maximal_words: ndarray[uint64], maximal_cliques: list[array[uint64]], max_cliques_as_words: ndarray[uint64]) -> ndarray[uint64]
*/
static PyObject *py_intersections_work(PyObject *self, PyObject *args)
{
    PyObject *max_obj, *cliques_list, *clique_words_obj;
    if (!PyArg_ParseTuple(args, "OOO",
                          &max_obj,
                          &cliques_list,
                          &clique_words_obj))
        return NULL;

    /* ---- unpack maximal_words ------------------------------------------------ */
    PyArrayObject *max_arr =
        (PyArrayObject *)PyArray_FROM_OTF(max_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!max_arr) return NULL;

    const uint64_t *max_words = (const uint64_t *)PyArray_DATA(max_arr);
    npy_uintp m = PyArray_SIZE(max_arr);
    if (m == 0 || m > 64) {
        Py_DECREF(max_arr);
        return PyErr_Format(PyExc_ValueError,
                            "maximal_words length must be 1..64 (got %zu)", (size_t)m);
    }
    const uint64_t full_mask = (m == 64) ? UINT64_MAX
                                         : ((UINT64_C(1) << m) - 1);

    /* ---- unpack max_cliques_as_words ---------------------------------------- */
    PyArrayObject *cw_arr =
        (PyArrayObject *)PyArray_FROM_OTF(clique_words_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
    if (!cw_arr) { Py_DECREF(max_arr); return NULL; }

    const uint64_t *clique_words = (const uint64_t *)PyArray_DATA(cw_arr);
    npy_uintp n_cliques = PyArray_SIZE(cw_arr);

    /* ---- validate maximal_cliques list length -------------------------------- */
    if (!PyList_CheckExact(cliques_list) ||
        PyList_GET_SIZE(cliques_list) != (Py_ssize_t)n_cliques) {
        Py_DECREF(max_arr); Py_DECREF(cw_arr);
        return PyErr_Format(PyExc_ValueError,
                            "maximal_cliques list length does not match clique_words");
    }

    /* ---- dynamic arrays for intersections & minimal non-faces ---------------- */
    uint64_t *intersections = NULL; size_t n_inter = 0, cap_inter = 0;
    uint64_t *mnf           = NULL; size_t n_mnf   = 0, cap_mnf   = 0;

    /* ---- helper lambda: append to dynamic array ------------------------------ */
#define APPEND(arr, n, cap, val)                               \
    do {                                                       \
        if ((n) == (cap)) {                                    \
            size_t new_cap = (cap ? cap * 2 : 32);             \
            uint64_t *tmp =                                  \
                (uint64_t *)realloc(arr, new_cap * sizeof(uint64_t)); \
            if (!tmp) { PyErr_NoMemory(); goto fail; }         \
            arr = tmp; cap = new_cap;                          \
        }                                                      \
        arr[(n)++] = (val);                                    \
    } while (0)

    /* ---- iterate cliques ----------------------------------------------------- */
    for (npy_uintp c_idx = 0; c_idx < n_cliques; ++c_idx) {
        uint64_t clique_word = clique_words[c_idx];
        uint64_t compl_word  = (~clique_word) & full_mask;

        /* ---- extract the numpy array of indices for this clique ------------- */
        PyObject *clq_obj = PyList_GET_ITEM(cliques_list, (Py_ssize_t)c_idx);
        PyArrayObject *clq_arr =
            (PyArrayObject *)PyArray_FROM_OTF(clq_obj, NPY_UINT64, NPY_ARRAY_IN_ARRAY);
        if (!clq_arr) { goto fail; }

        npy_intp clq_len = PyArray_SIZE(clq_arr);
        const uint64_t *clq_idx = (const uint64_t *)PyArray_DATA(clq_arr);
        if (clq_len < 2 || clq_len > 64) {
            Py_DECREF(clq_arr); PyErr_SetString(PyExc_ValueError,
               "each clique must have size between 2 and 64"); goto fail;
        }

        /* precompute shifts for clique vertices */
        uint64_t shifts[64];
        for (npy_intp i = 0; i < clq_len; ++i)
            shifts[i] = UINT64_C(1) << clq_idx[i];

        /* build *relevant* minimal non-faces list for this clique */
        uint64_t *rel_mnf = NULL; size_t n_rel = 0, cap_rel = 0;
        for (size_t i = 0; i < n_mnf; ++i)
            if ((mnf[i] & compl_word) == 0ULL)
                APPEND(rel_mnf, n_rel, cap_rel, mnf[i]);

        /* ---- enumerate k-subsets of clique (k >= 2) ------------------------- */
        for (npy_intp k = 2; k <= clq_len; ++k) {
            /* combination indices */
            npy_intp idx[64];
            for (npy_intp i = 0; i < k; ++i) idx[i] = i;
            int done = 0; size_t n_cand = 0, n_rej = 0;

            while (!done) {
                /* build mask for current subset */
                uint64_t mask = 0ULL;
                for (npy_intp j = 0; j < k; ++j) mask |= shifts[idx[j]];

                /* skip subsets containing any rel_mnf */
                int skip = 0;
                for (size_t r = 0; r < n_rel; ++r)
                    if ((mask & rel_mnf[r]) == rel_mnf[r]) { skip = 1; break; }

                if (!skip) {
                    ++n_cand;
                    int dummy_err = 0;
                    uint64_t inter = intersect_with_words(mask, max_words, m, &dummy_err);

                    if (inter) {
                        APPEND(intersections, n_inter, cap_inter, inter);
                    } else {
                        APPEND(mnf, n_mnf, cap_mnf, mask);
                        APPEND(rel_mnf, n_rel, cap_rel, mask);
                        ++n_rej;
                    }
                }

                /* next combination (lexicographic) */
                npy_intp p = k - 1;
                while (p >= 0 && idx[p] == clq_len - k + p) --p;
                if (p < 0) done = 1;
                else {
                    ++idx[p];
                    for (npy_intp j = p + 1; j < k; ++j)
                        idx[j] = idx[j - 1] + 1;
                }
            }
            if (n_cand == 0 || n_rej == n_cand) break; /* early exit rules */
        }
        free(rel_mnf);
        Py_DECREF(clq_arr);
    }

    /* ---- deduplicate & sort intersections ----------------------------------- */
    if (n_inter) {
        qsort(intersections, n_inter, sizeof(uint64_t), cmp_u64);
        size_t uniq = 1;
        for (size_t i = 1; i < n_inter; ++i)
            if (intersections[i] != intersections[uniq - 1])
                intersections[uniq++] = intersections[i];
        n_inter = uniq;
    }

    /* ---- package result as NumPy array -------------------------------------- */
    npy_intp dims[1] = {(npy_intp)n_inter};
    PyObject *out = PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!out) goto fail;
    if (n_inter)
        memcpy(PyArray_DATA((PyArrayObject *)out),
               intersections, n_inter * sizeof(uint64_t));

    /* ---- cleanup & return --------------------------------------------------- */
    free(intersections); free(mnf);
    Py_DECREF(max_arr); Py_DECREF(cw_arr);
    return out;

fail:
    free(intersections); free(mnf);
    Py_DECREF(max_arr); Py_DECREF(cw_arr);
    return NULL;
#undef APPEND
}
/* ========================================================================= */





static PyMethodDef Methods[] = {
    {"generate_increasing_tuples",         py_generate_increasing_tuples,         METH_VARARGS, "Generate all increasing k-tuples"},
    {"custom_bit_length",                 py_custom_bit_length,                 METH_VARARGS, "Bit length of unsigned int"},
    {"x_is_a_superset_of_any_in_list",    py_x_is_superset_of_any,              METH_VARARGS, "Return True if x supersets any b in list"},
    {"intersection_of_codewords_from_bits", py_intersection_of_codewords,        METH_VARARGS, "AND-intersect selected codewords"},
    {"lattice_slice2",                    py_lattice_slice2,                    METH_VARARGS, "Return uint64 ndarray of valid k-sets"},
    {"intersections_inside_a_clique",      py_intersections_inside_a_clique,     METH_VARARGS, "Return unique intersections inside a clique"},
    {"intersections_work", py_intersections_work, METH_VARARGS, "Compute all non-maximal intersections"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "translated_functions",
    "C translations of performance-critical helpers",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_translated_functions(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    
    /* Import numpy C API */
    import_array();
    if (PyErr_Occurred()) {
        Py_DECREF(module);
        return NULL;
    }
    
    return module;
}

