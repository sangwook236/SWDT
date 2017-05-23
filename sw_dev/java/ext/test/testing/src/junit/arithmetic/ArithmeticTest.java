package junit.arithmetic;

import junit.framework.TestCase;

public class ArithmeticTest extends TestCase {

	public void setUp() throws Exception {
		x_ = 3.0;
		y_ = 2.0;
		System.out.println("ArithmeticTest.setUp() is called");
	}

	public void tearDown() throws Exception {
		System.out.println("ArithmeticTest.tearDown() is called");
	}

	public void testAdd() {
		final double r = arithmetic.Arithmetic.add(x_, y_);
		assertEquals(5.0, r, 1.0e-10);
	}

	public void testSub() {
		final double r = arithmetic.Arithmetic.sub(x_, y_);
		assertEquals(1.0, r, 1.0e-10);
	}

	public void testMul() {
		final double r = arithmetic.Arithmetic.mul(x_, y_);
		assertEquals(6.0, r, 1.0e-10);
	}

	public void testDiv() {
		final double r = arithmetic.Arithmetic.div(x_, y_);
		assertEquals(1.5, r, 1.0e-10);
	}

	public void testDivideByZero() {
		try
		{
			arithmetic.Arithmetic.divideByZero();
			fail("Divided by Zero!");
		}
		catch (ArithmeticException e)
		{
			assertNotNull(e.getMessage());
		}
		finally
		{
		}
	}

	private double x_;
	private double y_;
}
