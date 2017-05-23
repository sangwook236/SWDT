
public class PrimitiveDataType {

	private static void processInt()
	{
		int i1 = 10;
		int i2 = 10;

		System.out.println("i1 = " + i1);
		System.out.println("i2 = " + i2);
		//System.out.println("i1 equals i2: " + i1.equals(i2));
		System.out.println("i1 == i2: " + (i1 == i2));

		//
		int i3 = i1;

		System.out.println("i1 = " + i1);
		System.out.println("i3 = " + i3);
		//System.out.println("i1 equals i3: " + i1.equals(i3));
		System.out.println("i1 == i3: " + (i1 == i3));

		i3 = 20;

		System.out.println("i1 = " + i1);
		System.out.println("i3 = " + i3);
		//System.out.println("i1 equals i3: " + i1.equals(i3));
		System.out.println("i1 == i3: " + (i1 == i3));
	}

	private static void processString()
	{
		String str1 = new String("abc");
		String str2 = new String("abc");

		System.out.println("str1 = " + str1);
		System.out.println("str2 = " + str2);
		System.out.println("str1 equals str2: " + str1.equals(str2));
		System.out.println("str1 == str2: " + (str1 == str2));

		String str3 = str1;

		System.out.println("str1 = " + str1);
		System.out.println("str3 = " + str3);
		System.out.println("str1 equals str3: " + str1.equals(str3));
		System.out.println("str1 == str3: " + (str1 == str3));

		str3 = "def";

		System.out.println("str1 = " + str1);
		System.out.println("str3 = " + str3);
		//System.out.println("str1 equals str3: " + str1.equals(str3));
		//System.out.println("str1 == str3: " + (str1 == str3));
	}

	private static void processObject()
	{
		MyInteger aMyInt1 = new MyInteger(-10);
		MyInteger aMyInt2 = new MyInteger(-10);

		System.out.println("aMyInt1 = " + aMyInt1.get());
		System.out.println("aMyInt2 = " + aMyInt2.get());
		System.out.println("aMyInt1 equals aMyInt2: " + aMyInt1.equals(aMyInt2));
		System.out.println("aMyInt1 == aMyInt2: " + (aMyInt1 == aMyInt2));

		MyInteger aMyInt3 = aMyInt1;

		System.out.println("aMyInt1 = " + aMyInt1.get());
		System.out.println("aMyInt3 = " + aMyInt3.get());
		System.out.println("aMyInt1 equals aMyInt3: " + aMyInt1.equals(aMyInt3));
		System.out.println("aMyInt1 == aMyInt3: " + (aMyInt1 == aMyInt3));

		aMyInt3.set(-100);

		System.out.println("aMyInt1 = " + aMyInt1.get());
		System.out.println("aMyInt3 = " + aMyInt3.get());
		System.out.println("aMyInt1 equals aMyInt3: " + aMyInt1.equals(aMyInt3));
		System.out.println("aMyInt1 == aMyInt3: " + (aMyInt1 == aMyInt3));
	}

	static void runAll()
	{
		processInt();
		System.out.println();
		processString();
		System.out.println();
		processObject();
	}

}

class MyInteger
{
	public MyInteger(int i)
	{
		i_ = i;
	}

	public void set(int i)
	{
		i_ = i;
	}

	public int get()
	{
		return i_;
	}

	int i_;
}
