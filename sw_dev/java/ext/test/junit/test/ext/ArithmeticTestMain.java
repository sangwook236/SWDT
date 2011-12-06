package ext;

import junit.framework.Test;
import junit.framework.TestSuite;

public final class ArithmeticTestMain {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		junit.textui.TestRunner.run(suite());
	}

	public static Test suite()
	{
		TestSuite suite = new TestSuite("All Arithmetic Tests");
		suite.addTestSuite(ArithmeticSimpleTest.class);
		suite.addTestSuite(ArithmeticTest.class);
		return suite;
	}
}
