import java.util.*;

class HelloJniClass
{
	native void Hello();

	static
	{
		System.loadLibrary("Hello_DLL");
	}

	public static void main(String args[])
	{
		HelloJniClass myJNI = new HelloJniClass();
		myJNI.Hello();
	}
}
