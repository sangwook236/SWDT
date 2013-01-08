
public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			javacpp.JavaCPPMain.run(args);
			//jni.JNIMain.run(args);  // not yet implemented
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
