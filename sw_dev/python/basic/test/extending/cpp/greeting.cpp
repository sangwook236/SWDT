#include <Python.h>
#include <iostream>


// Usage in Python:
//	import greeting
//	status = greeting.greet("ls -l")

static PyObject * greet(PyObject *self, PyObject *args)
{
	std::cout << "hello, world" << std::endl;
	Py_INCREF(Py_None);
	return Py_None;
}

// Method setting.
static PyMethodDef greeting_methods[] = {
	{ "greet", greet, METH_VARARGS, "greet() doc string" },
	{ NULL, NULL }
};

// Module setting.
static struct PyModuleDef greetingmodule = {
	PyModuleDef_HEAD_INIT,
	"greeting",
	"greeting module doc string",
	-1,
	greeting_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_greeting(void)  // This name has to agree with the module name in setup.py.
{
	return PyModule_Create(&greetingmodule);
}
