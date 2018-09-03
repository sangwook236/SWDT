package jython;

import org.python.util.PythonInterpreter;
import org.python.core.PyInteger;
import org.python.core.PyFunction;
import java.util.Properties;

public class Jython_Main {

	// NOTE [caution] {important} >> Touch and rebuild this file to reload an edited Python script.

	public static void run(String[] args)
	{
		Properties props = new Properties();
		props.put("python.home", "D:/util/Anaconda3/envs/ml_py3");
		props.put("python.console.encoding", "UTF-8");  // Use to prevent: console: Failed to install '': java.nio.charset.UnsupportedCharsetException: cp0.
		props.put("python.security.respectJavaAccessibility", "false");  // Don't respect java accessibility, so that we can access protected members on subclasses.
		props.put("python.import.site", "false");

		Properties preprops = System.getProperties();
		//for (Object key : preprops.keySet())
		//	System.out.println(key + " = " + preprops.get(key));
    			
		PythonInterpreter.initialize(preprops, props, new String[] {""});
        PythonInterpreter interpreter = new PythonInterpreter();

        // Append additional paths.
		interpreter.exec(
			"import sys\n" +
			"sys.path.append('D:/work/SWDT_github/sw_dev/java/ext/test/native_interface/src/python')\n"
		);
		//interpreter.exec("import sys; print('sys.path =', sys.path)");

        //
		runSimpleExample(interpreter);
		runBuiltInFunction(interpreter);

		runObjectFactoryExample(interpreter);
	}
	
	// REF [site] >> https://smartbear.com/blog/test-and-monitor/embedding-jython-in-java-applications/
	private static void runSimpleExample(PythonInterpreter interpreter)
	{
		interpreter.set("val", new PyInteger(42));
		interpreter.exec("square = val * val");
		PyInteger sqr = (PyInteger)interpreter.get("square");
		System.out.println("42^2 = " + sqr);

		interpreter.exec("from arithmetic import add, sub");
		interpreter.set("val1", new PyInteger(12));
		interpreter.set("val2", new PyInteger(25));
		interpreter.exec("result = add(val1, val2)");
		interpreter.exec("print('{} + {} = {}'.format(val1, val2, result))");
		PyInteger result = (PyInteger)interpreter.get("result");
		System.out.println("Result: " + result.asInt());

		PyFunction sub_func = (PyFunction)interpreter.get("sub");
		final double sub_result = sub_func.__call__(new PyInteger(5), new PyInteger(2)).asDouble();
		System.out.println("5 - 2 = " + sub_result);
	}
	
	private static void runBuiltInFunction(PythonInterpreter interpreter)
	{
		//interpreter.exec("import sys, os");
		//interpreter.exec("import numpy");
	}

	// REF [site] >> http://www.jython.org/jythonbook/en/1.0/JythonAndJavaIntegration.html
	private static void runObjectFactoryExample(PythonInterpreter interpreter)
	{
		BuildingFactory factory = new BuildingFactory(interpreter);

		IBuilding building1 = factory.create("BUILDING-A", "100 WEST MAIN", "1");
		System.out.println(building1.toString());
		print(building1);
		print(factory.create("BUILDING-B", "110 WEST MAIN", "2"));
		print(factory.create("BUILDING-C", "120 WEST MAIN", "3"));
	}
	
	private static void print(IBuilding building)
	{
		System.out.println("Building Info: " +
			building.getBuildingId() + " " +
			building.getBuildingName() + " " +
			building.getBuildingAddress()
		);
    }

}
