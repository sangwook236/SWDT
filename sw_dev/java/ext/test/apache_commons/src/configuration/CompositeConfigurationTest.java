package configuration;

import org.apache.commons.configuration2.*;
import org.apache.commons.configuration2.ex.ConfigurationException;
import org.apache.commons.configuration2.builder.FileBasedConfigurationBuilder;
import org.apache.commons.configuration2.builder.fluent.Parameters;
import org.apache.commons.configuration2.convert.DefaultListDelimiterHandler;

final class CompositeConfigurationTest {

	public static void run(String[] args)
	{
		try
		{
			CompositeConfiguration config = new CompositeConfiguration();

			config.addConfiguration(new SystemConfiguration());

			FileBasedConfigurationBuilder<PropertiesConfiguration> builder =
				    new FileBasedConfigurationBuilder<PropertiesConfiguration>(PropertiesConfiguration.class)
				    .configure(new Parameters().properties()
				        .setFileName("src/configuration/user_gui.properties")
				        .setThrowExceptionOnMissing(true)
				        .setListDelimiterHandler(new DefaultListDelimiterHandler(';'))
				        .setIncludesAllowed(false));
			config.addConfiguration(builder.getConfiguration());
		}
		catch (ConfigurationException ex)
		{
			ex.printStackTrace();
		}
		finally
		{
		}
	}

}
