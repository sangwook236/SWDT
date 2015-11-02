package junit;

import junit.framework.Test;
import junit.framework.TestSuite;

public final class JUnit_Main {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		junit.textui.TestRunner.run(suite());
	}

	public static Test suite()
	{
		TestSuite suite = new TestSuite("All Arithmetic Tests");
		suite.addTestSuite(junit.arithmetic.ArithmeticSimpleTest.class);
		suite.addTestSuite(junit.arithmetic.ArithmeticTest.class);
		return suite;
	}
}
