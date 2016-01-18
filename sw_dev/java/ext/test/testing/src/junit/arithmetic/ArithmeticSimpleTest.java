package junit.arithmetic;

import junit.framework.TestCase;

/**
 * @author sangwook
 *
 */
public class ArithmeticSimpleTest extends TestCase {

	/**
	 * Test method for {@link ext.Arithmetic#add(double, double)}.
	 */
	public void testAdd() {
		final double r = arithmetic.Arithmetic.add(x_, y_);
		assertEquals(5.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#sub(double, double)}.
	 */
	public void testSub() {
		final double r = arithmetic.Arithmetic.sub(x_, y_);
		assertEquals(1.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#mul(double, double)}.
	 */
	public void testMul() {
		final double r = arithmetic.Arithmetic.mul(x_, y_);
		assertEquals(6.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#div(double, double)}.
	 */
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

	private double x_ = 3.0;
	private double y_ = 2.0;
}
