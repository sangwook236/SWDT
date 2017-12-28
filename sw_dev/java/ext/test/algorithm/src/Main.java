
public class Main {

	public static void main(String[] args) {
		try
		{
			System.out.println("Apache Lucene project -----------------------------------------------");
			//lucene.Lucene_Main.run(args);  // Not yet implemented.

			System.out.println("Apache Solr project -------------------------------------------------");
			solr.Solr_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
