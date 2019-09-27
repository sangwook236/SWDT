
public class Main {

	public static void main(String[] args) {
		try
		{
			System.out.println("Apache OpenNLP ------------------------------------------------------");			
			opennlp.OpenNLP_Main.run(args);  // Not yet implemented.

			System.out.println("Apache Tika ---------------------------------------------------------");			
			tika.Tika_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
