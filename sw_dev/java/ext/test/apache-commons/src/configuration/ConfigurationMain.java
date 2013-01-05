package configuration;

public final class ConfigurationMain {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		PropertiesConfigurationTest.run(args);
		XmlConfigurationTest.run(args);
		CompositeConfigurationTest.run(args);
	}

}
