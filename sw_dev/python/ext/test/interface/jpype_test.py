#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import jpype
import jpype.imports

def basic_example():
	class_path = jpype.getClassPath()
	print("Class path: {}.".format(class_path))
	jvm_path = jpype.getDefaultJVMPath()
	print("JVM path: {}.".format(jvm_path))

	if not jpype.isJVMStarted():
		try:
			#jpype.addClassPath("/path/to/sample.jar")
			#jpype.startJVM(jvmpath=jvmpath, classpath=classpath, ignoreUnrecognized=False, convertStrings=False, interrupt=False)
			jpype.startJVM()
		except TypeError as ex:
			print("TypeError raised: {}.".format(ex))
		except OSError as ex:
			print("OSError raised: {}.".format(ex))

	#--------------------
	# Do something.

	#--------------------
	if jpype.isJVMStarted():
		jpype.shutdownJVM()

# REF [site] >> https://github.com/jpype-project/jpype/blob/master/examples/rmi.py
def rmi_example():
	if not jpype.isJVMStarted():
		try:
			jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=/path/to/classes")
		except TypeError as ex:
			print("TypeError raised: {}.".format(ex))
		except OSError as ex:
			print("OSError raised: {}.".format(ex))

	#--------------------
	import java.rmi

	try:
		p = java.rmi.Naming.lookup("rmi://localhost:2004/server")
	except java.rmi.ConnectException as ex:
		print("java.rmi.ConnectException raised: {}.".format(ex))
		return
	print(p, p.__class__)

	p.callRemote()

	#--------------------
	if jpype.isJVMStarted():
		jpype.shutdownJVM()

def main():
	# REF [file] >> ${SWDT_PYTHON_HOME}/ext/test/documentation/pdfbox_test.py

	basic_example()
	#rmi_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
