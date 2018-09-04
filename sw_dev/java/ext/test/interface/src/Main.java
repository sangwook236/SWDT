
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("JavaCPP -------------------------------------------------------------");
			//javacpp.JavaCPP_Main.run(args);

			System.out.println("Java Native Interface (JNI) -----------------------------------------");
			//jni.JNIMain.run(args);  // Not yet implemented.

			System.out.println("Jython --------------------------------------------------------------");
			jython.Jython_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
