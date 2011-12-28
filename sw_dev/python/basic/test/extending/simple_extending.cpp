#include "stdafx.h"
#include <Python.h>


static PyObject * spam_system(PyObject *self, PyObject *args)
{
	const char *command;
	if (!PyArg_ParseTuple(args, "s", &command))
		return NULL;

	const int sts = system(command);

	return Py_BuildValue("i", sts);
}

/*
in python interperter:
	import spam
	status = spam.system("ls -l")
*/
