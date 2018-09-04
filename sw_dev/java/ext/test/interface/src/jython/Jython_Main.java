package jython;

import org.python.util.PythonInterpreter;
import org.python.core.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Jython_Main {

	// NOTE [caution] {important} >> Touch and rebuild this file to reload an edited Python script.

	public static void run(String[] args)
	{
		initializeJythonRuntime("src/jython/jython_config.properties");

 		runSimpleExample();
		runBuiltInModule();

		runObjectFactoryExample();
	}

	private static void initializeJythonRuntime(String configPath)
	{
		Properties props = new Properties();
		try
		{
			InputStream stream = null;
			try
			{
				stream = new FileInputStream(configPath);
			} 
			catch (IOException ex)
			{
				System.err.println("File not found: " + configPath + '.');
				//ex.printStackTrace();
				throw ex;
			}
			finally
			{
				if (null != stream)
				{
					try
					{
						props.load(stream);
						stream.close();
					}
					catch (IOException ex)
					{
						System.err.println("Failed to load Jython configurations.");
						//ex.printStackTrace();
						throw ex;
					}
				}
			}
		}
		catch (Exception ex)
		{
		}
		finally
		{
			// REF [site] >> https://wiki.python.org/jython/UserGuide#the-jython-registry
			//final String PYTHON_HOME = "D:/util/Anaconda3/envs/ml_py3";
			//props.put("python.home", PYTHON_HOME);
			//props.put("python.path", PYTHON_HOME + "/Lib/site-packages");
			props.put("python.console.encoding", "UTF-8");  // Use to prevent: console: Failed to install '': java.nio.charset.UnsupportedCharsetException: cp0.
			props.put("python.security.respectJavaAccessibility", "false");  // Don't respect java accessibility, so that we can access protected members on subclasses.
			props.put("python.import.site", "false");
		}

		Properties preprops = System.getProperties();
		//for (Object key : preprops.keySet())
		//	System.out.println(key + " = " + preprops.get(key));

		// Initializes the Jython runtime.
		// This should only be called once, before any other Python objects (including PythonInterpreter) are created.
		PythonInterpreter.initialize(preprops, props, new String[] {""});
	}
	
	// REF [site] >> https://smartbear.com/blog/test-and-monitor/embedding-jython-in-java-applications/
	private static void runSimpleExample()
	{
		try (PythonInterpreter interpreter = new PythonInterpreter())
		{
			// Append additional paths.
			interpreter.exec("import sys");
			interpreter.eval("sys.path.append('src/python')");
			//interpreter.exec(
			//	"import sys\n" +
			//	"sys.path.append('src/python')\n"  // For arithmetic.py.
			//);

			//final PyObject sys_path = interpreter.eval("sys.path");
			//System.out.println("sys.path = " + sys_path);
	
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
			
			interpreter.close();
		}
		catch (Exception ex)
		{
			ex.printStackTrace();
		}
	}

	private static void runBuiltInModule()
	{
		try (PythonInterpreter interpreter = new PythonInterpreter())
		{
			/*
			// TODO [enhance] >> How to get Python home path?
			//	Use python.path in jython_config.properties.
			PyList paths = interpreter.getSystemState().path;
			final String pythonHomeLibPath = paths.get(0).toString();
	
			// Append additional paths.
			interpreter.exec(
				"import sys\n" +
				"sys.path.append('" + pythonHomeLibPath + "/site-packages')\n"  // For Python modules like numpy.
			);
			*/
	
			//interpreter.exec("import sys, os");
			interpreter.exec("import numpy");
			
			interpreter.close();
		}
		catch (Exception ex)
		{
			ex.printStackTrace();
		}
	}

	// REF [site] >> http://www.jython.org/jythonbook/en/1.0/JythonAndJavaIntegration.html
	private static void runObjectFactoryExample()
	{
		try (PythonInterpreter interpreter = new PythonInterpreter())
		{
			BuildingFactory factory = new BuildingFactory(interpreter);
	
			IBuilding building1 = factory.create("BUILDING-A", "100 WEST MAIN", "1");
			System.out.println(building1.toString());
			print(building1);
			print(factory.create("BUILDING-B", "110 WEST MAIN", "2"));
			print(factory.create("BUILDING-C", "120 WEST MAIN", "3"));
			
			interpreter.close();
		}
		catch (Exception ex)
		{
			ex.printStackTrace();
		}
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
