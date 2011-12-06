package ext;

import org.apache.commons.configuration.*;

public final class CompositeConfigurationTest {
	public static void runAll()
	{
		try
		{
			CompositeConfiguration config = new CompositeConfiguration();
			
			config.addConfiguration(new SystemConfiguration());
			config.addConfiguration(new PropertiesConfiguration("data/user_gui.properties"));
		}
		catch (ConfigurationException e)
		{
			e.printStackTrace();
		}
		finally
		{
		}
	}
}
