#include <Python.h>
#include <iostream>


void embedding_simple_string()
{
	Py_Initialize();
	
	PyRun_SimpleString(
		"from time import time,ctime\n"
		"print(\'Today is\', ctime(time()))\n"
	);
	
	Py_Finalize();
}
