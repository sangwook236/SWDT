package jython;

import org.python.core.PyObject;
import org.python.core.PyString;
import org.python.util.PythonInterpreter;

/**
 * Object Factory that is used to coerce Python module into a Java class.
 */
public class BuildingFactory {

	/**
	 * Create a new PythonInterpreter object, then use it to execute some Python code.
	 * In this case, we want to import the Python module that we will coerce.
	 *
	 * Once the module is imported than we obtain a reference to it and assign the reference to a Java variable.
	 */
	public BuildingFactory(PythonInterpreter interpreter)
	{
		interpreter.exec("from Building import Building");
		buildingClass_ = interpreter.get("Building");  // Get the building class.
    }

	/**
	 * The create method is responsible for performing the actual coercion of the referenced Python module into Java bytecode.
	 */
	public IBuilding create(String name, String location, String id)
	{
		PyObject buildingObject = buildingClass_.__call__(new PyString(name), new PyString(location), new PyString(id));  // Call a constructor.
		return (IBuilding)buildingObject.__tojava__(IBuilding.class);
	}

	private PyObject buildingClass_;

}
