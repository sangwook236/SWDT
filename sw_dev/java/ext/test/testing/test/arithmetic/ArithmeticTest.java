package arithmetic;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ArithmeticTest {

    @BeforeAll
    static void initAll() {
    }

    @BeforeEach
	void init() throws Exception {
		x_ = 3.0;
		y_ = 2.0;
		System.out.println("ArithmeticTest.setUp() is called");
	}

	@AfterEach
	void tearDown() throws Exception {
		System.out.println("ArithmeticTest.tearDown() is called");
	}

	@AfterAll
    static void tearDownAll() {
    }

	@Test
	void testAdd() {
		final double r = arithmetic.Arithmetic.add(x_, y_);
		assertEquals(5.0, r, 1.0e-10);
	}

	@Test
	void testSub() {
		final double r = arithmetic.Arithmetic.sub(x_, y_);
		assertEquals(1.0, r, 1.0e-10);
	}

	@Test
	void testMul() {
		final double r = arithmetic.Arithmetic.mul(x_, y_);
		assertEquals(6.0, r, 1.0e-10);
	}

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

	private double x_;
	private double y_;
}
