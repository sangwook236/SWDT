#include <Python.h>
#include <iostream>
#include <cstring>


#if defined(_UNICODE) || defined(UNICODE)
bool embedding_simple_script(int argc, wchar_t* argv[])
#else
bool embedding_simple_script(int argc, char* argv[])
#endif
{
	if (argc < 3)
	{
		std::cerr << "Usage: call pythonfile funcname [args]" << std::endl;
		std::cerr << "\te.g.) embedding.exe arithmetic add 3 2" << std::endl;
		return false;
	}

	Py_Initialize();
#if defined(_UNICODE) || defined(UNICODE)
	PyObject *pName = PyUnicode_FromUnicode(argv[1], std::wcslen(argv[1]));
#else
	//PyObject *pName = PyString_FromString(argv[1]);
	PyObject *pName = PyUnicode_FromString(argv[1]);
#endif
	// Error checking of pName left out

	PyObject *pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	if (pModule != NULL)
	{
#if defined(_UNICODE) || defined(UNICODE)
		PyObject *pTmp = PyUnicode_FromWideChar(argv[2], std::wcslen(argv[2]));
		PyObject *pFunc = PyObject_GetAttr(pModule, pTmp);
		Py_DECREF(pTmp);
#else
		PyObject *pFunc = PyObject_GetAttrString(pModule, argv[2]);
#endif
		// pFunc is a new reference

		if (pFunc && PyCallable_Check(pFunc))
		{
			PyObject *pValue = NULL;

			PyObject *pArgs = PyTuple_New(argc - 3);
			for (int i = 0; i < argc - 3; ++i)
			{
				// TODO [check] >> is it correct?
#if defined(_UNICODE) || defined(UNICODE)
				pValue = PyLong_FromUnicode(argv[i + 3], std::wcslen(argv[i + 3]), 0);
#else
				pValue = PyLong_FromLong(atoi(argv[i + 3]));
				//pValue = PyLong_FromString(argv[i + 3], NULL, 0);
#endif
				if (!pValue)
				{
					Py_DECREF(pArgs);
					Py_DECREF(pModule);
					std::cerr << "Cannot convert argument" << std::endl;
					return false;
				}

				// pValue reference stolen here
				PyTuple_SetItem(pArgs, i, pValue);
			}

			pValue = PyObject_CallObject(pFunc, pArgs);
			Py_DECREF(pArgs);

			if (pValue != NULL)
			{
				std::cout << "Result of call: " << PyLong_AsLong(pValue) << std::endl;
				Py_DECREF(pValue);
			}
			else
			{
				Py_DECREF(pFunc);
				Py_DECREF(pModule);

				PyErr_Print();
				std::cerr << "Call failed" << std::endl;
				return false;
			}
		}
		else
		{
			if (PyErr_Occurred())
				PyErr_Print();
			std::cerr << "Cannot find function \"" << argv[2] << "\"" << std::endl;
		}

		Py_XDECREF(pFunc);
		Py_DECREF(pModule);
	}
	else
	{
		PyErr_Print();
		std::cerr << "Failed to load \"" << argv[1] << "\"" << std::endl;
		return false;
	}

	Py_Finalize();
	return true;
}
