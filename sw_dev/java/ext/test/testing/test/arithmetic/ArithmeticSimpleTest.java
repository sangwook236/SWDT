package arithmetic;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * @author sangwook
 *
 */
public class ArithmeticSimpleTest {

	/**
	 * Test method for {@link ext.Arithmetic#add(double, double)}.
	 */
	@Test
	void testAdd() {
		final double r = arithmetic.Arithmetic.add(x_, y_);
		assertEquals(5.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#sub(double, double)}.
	 */
	@Test
	void testSub() {
		final double r = arithmetic.Arithmetic.sub(x_, y_);
		assertEquals(1.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#mul(double, double)}.
	 */
	@Test
	void testMul() {
		final double r = arithmetic.Arithmetic.mul(x_, y_);
		assertEquals(6.0, r, 1.0e-10);
	}

	/**
	 * Test method for {@link ext.Arithmetic#div(double, double)}.
	 */
	@Test
	void testDiv() {
		final double r = arithmetic.Arithmetic.div(x_, y_);
		assertEquals(1.5, r, 1.0e-10);
	}

	@Test
	void testDivideByZero() {
		try
		{
			arithmetic.Arithmetic.divideByZero();
			fail("Divided by Zero!");
		}
		catch (ArithmeticException ex)
		{
			assertNotNull(ex.getMessage());
		}
		finally
		{
		}
	}

	private double x_ = 3.0;
	private double y_ = 2.0;
}
