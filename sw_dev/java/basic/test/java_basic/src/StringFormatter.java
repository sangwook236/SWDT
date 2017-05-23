import java.util.Formatter;
import java.util.Locale;
import java.util.Calendar;
import java.util.GregorianCalendar;
import static java.util.Calendar.*;

public class StringFormatter {
	public static void runAll()
	{
		//
		System.out.printf("%1$s%n%2$s%n%3$s%n", 10, "20", 30.0);

		final Calendar c = new GregorianCalendar(1995, MAY, 23);
		String str = String.format("Duke's Birthday: %1$tm %1$te,%1$tY", c);		
		System.out.println(str);

		//
		StringBuilder sb = new StringBuilder();
		// Send all output to the appendable object sb
		Formatter formatter = new Formatter(sb, Locale.US);

		// Explicit argument indices may be used to re-order output.
		formatter.format("%4$2s %3$2s %2$2s %1$2s", "a", "b", "c", "d");
		System.out.println(formatter);

		// Optional locale as the first argument can be used to get locale-specific formatting of numbers.
		// The precision and width can be given to round and align the value.
		formatter.format(Locale.FRANCE, "e = %+10.4f", Math.E);
		System.out.println(formatter);

		// The '(' numeric flag may be used to format negative numbers with parentheses rather than a minus sign.
		// Group separators are automatically inserted.
		final double balanceDelta = -6217.58;
		formatter.format("Amount gained or lost since last statement: $ %(,.2f", balanceDelta);
		System.out.println(formatter);
	}
}
