
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			javacpp.JavaCPP_Main.run(args);
			//jni.JNIMain.run(args);  // not yet implemented
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
