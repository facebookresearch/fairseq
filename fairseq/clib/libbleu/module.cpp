/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Python.h>


static PyMethodDef method_def[] = {
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
   PyModuleDef_HEAD_INIT,
   "libbleu",   /* name of module */
   NULL,     /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   method_def
};


#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_libbleu()
#else
PyMODINIT_FUNC PyInit_libbleu()
#endif
{
  PyObject *m = PyModule_Create(&module_def);
  if (!m) {
    return NULL;
  }
  return m;
}
