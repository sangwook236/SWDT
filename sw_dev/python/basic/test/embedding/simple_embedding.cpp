#include <Python.h>
#include <iostream>


void simple_embedding()
{
	Py_Initialize();
	
	PyRun_SimpleString(
		"from time import time,ctime\n"
		"print(¡¯Today is¡¯, ctime(time()))\n"
	);
	
	Py_Finalize();
}
