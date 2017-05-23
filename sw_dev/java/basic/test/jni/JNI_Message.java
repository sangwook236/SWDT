class JNI_Message
{
	native String Message(String input);

	// Load the library.
	static
	{
		System.loadLibrary("Msg_DLL");
	}

	public static void main(String args[])
	{
		String buf;

		// Create class instance.
		JNI_Message myJNI = new JNI_Message();

		// Call native method.
		buf = myJNI.Message("Apple");

		System.out.print(buf);
	}
}
