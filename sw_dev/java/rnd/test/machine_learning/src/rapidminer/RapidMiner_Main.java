package rapidminer;

public class RapidMiner_Main {

	public static void run(String[] args)
	{
		runPCASample();
	}

	private static void runPCASample()
	{
		RapidMinerPCA proc = new RapidMinerPCA();

		proc.loadData();
		proc.run();
	}

}
