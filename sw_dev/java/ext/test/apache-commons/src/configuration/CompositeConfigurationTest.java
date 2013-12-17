package configuration;

import org.apache.commons.configuration.*;

final class CompositeConfigurationTest {
	
	public static void run(String[] args)
	{
		try
		{
			CompositeConfiguration config = new CompositeConfiguration();
			
			config.addConfiguration(new SystemConfiguration());
			config.addConfiguration(new PropertiesConfiguration("src/configuration/user_gui.properties"));
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
