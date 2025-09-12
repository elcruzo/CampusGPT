#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "fast_tokenizer.h"

typedef struct {
    PyObject_HEAD
    Tokenizer* tokenizer;
} FastTokenizerObject;

static void FastTokenizer_dealloc(FastTokenizerObject* self) {
    if (self->tokenizer) {
        tokenizer_free(self->tokenizer);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* FastTokenizer_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    FastTokenizerObject* self;
    self = (FastTokenizerObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->tokenizer = NULL;
    }
    return (PyObject*)self;
}

static int FastTokenizer_init(FastTokenizerObject* self, PyObject* args, PyObject* kwds) {
    const char* vocab_file;
    
    if (!PyArg_ParseTuple(args, "s", &vocab_file)) {
        return -1;
    }

    self->tokenizer = tokenizer_load(vocab_file);
    if (!self->tokenizer) {
        PyErr_SetString(PyExc_IOError, "failed to load vocab file");
        return -1;
    }

    return 0;
}

static PyObject* FastTokenizer_encode(FastTokenizerObject* self, PyObject* args) {
    const char* text;
    
    if (!PyArg_ParseTuple(args, "s", &text)) {
        return NULL;
    }

    TokenizedText* result = tokenizer_encode(self->tokenizer, text);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "encoding failed");
        return NULL;
    }

    PyObject* token_list = PyList_New(result->length);
    if (!token_list) {
        tokenized_text_free(result);
        return NULL;
    }

    for (size_t i = 0; i < result->length; i++) {
        PyObject* token = PyLong_FromUnsignedLong(result->tokens[i]);
        PyList_SET_ITEM(token_list, i, token);
    }

    tokenized_text_free(result);
    return token_list;
}

static PyObject* FastTokenizer_decode(FastTokenizerObject* self, PyObject* args) {
    PyObject* token_list;
    
    if (!PyArg_ParseTuple(args, "O", &token_list)) {
        return NULL;
    }

    if (!PyList_Check(token_list)) {
        PyErr_SetString(PyExc_TypeError, "expected list of tokens");
        return NULL;
    }

    Py_ssize_t length = PyList_Size(token_list);
    uint32_t* tokens = malloc(length * sizeof(uint32_t));
    if (!tokens) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < length; i++) {
        PyObject* item = PyList_GetItem(token_list, i);
        tokens[i] = PyLong_AsUnsignedLong(item);
    }

    char* decoded = tokenizer_decode(self->tokenizer, tokens, length);
    free(tokens);

    if (!decoded) {
        PyErr_SetString(PyExc_RuntimeError, "decoding failed");
        return NULL;
    }

    PyObject* result = PyUnicode_FromString(decoded);
    free(decoded);
    return result;
}

static PyMethodDef FastTokenizer_methods[] = {
    {"encode", (PyCFunction)FastTokenizer_encode, METH_VARARGS, "encode text to tokens"},
    {"decode", (PyCFunction)FastTokenizer_decode, METH_VARARGS, "decode tokens to text"},
    {NULL}
};

static PyTypeObject FastTokenizerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "campusgpt_tokenizer.FastTokenizer",
    .tp_doc = "fast BPE tokenizer",
    .tp_basicsize = sizeof(FastTokenizerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = FastTokenizer_new,
    .tp_init = (initproc)FastTokenizer_init,
    .tp_dealloc = (destructor)FastTokenizer_dealloc,
    .tp_methods = FastTokenizer_methods,
};

static PyModuleDef campusgpt_tokenizer_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "campusgpt_tokenizer",
    .m_doc = "fast tokenizer for campusgpt",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_campusgpt_tokenizer(void) {
    PyObject* m;

    if (PyType_Ready(&FastTokenizerType) < 0)
        return NULL;

    m = PyModule_Create(&campusgpt_tokenizer_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&FastTokenizerType);
    if (PyModule_AddObject(m, "FastTokenizer", (PyObject*)&FastTokenizerType) < 0) {
        Py_DECREF(&FastTokenizerType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
