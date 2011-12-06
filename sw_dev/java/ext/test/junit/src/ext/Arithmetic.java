package ext;

public class Arithmetic {
	public static double add(final double lhs, final double rhs)
	{
		return lhs + rhs;
	}

	public static double sub(final double lhs, final double rhs)
	{
		return lhs - rhs;
	}

	public static double mul(final double lhs, final double rhs)
	{
		return lhs * rhs;
	}

	public static double div(final double lhs, final double rhs) throws ArithmeticException
	{
		return lhs / rhs;
	}
	
	public static double divideByZero() throws ArithmeticException
	{
		return 2 / 0;
	}
}
