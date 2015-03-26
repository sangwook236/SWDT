
public class Main {

	public static void main(String[] args) {
		try
		{
			System.out.println("HYDRA MCMC Library --------------------------------------------------");
			hydra_mcmc.HydraMCMC_Main.run(args);  // not yet implemented.
		}
		catch (Exception e)
		{
			System.err.println("Exception occurred: " + e.toString());
		}
	}

}
