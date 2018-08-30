package configuration;

import org.apache.commons.configuration2.*;
import org.apache.commons.configuration2.builder.FileBasedConfigurationBuilder;
import org.apache.commons.configuration2.builder.fluent.Parameters;
import org.apache.commons.configuration2.convert.DefaultListDelimiterHandler;
import org.apache.commons.configuration2.ex.*;
//import java.util.ArrayList;

final class XmlConfigurationTest {

	public static void run(String[] args)
	{
		try
		{
			FileBasedConfigurationBuilder<XMLConfiguration> builder =
				    new FileBasedConfigurationBuilder<XMLConfiguration>(XMLConfiguration.class)
				    .configure(new Parameters().properties()
				        .setFileName("src/configuration/user_gui.xml")
				        .setThrowExceptionOnMissing(true)
				        .setListDelimiterHandler(new DefaultListDelimiterHandler(';'))
				        .setIncludesAllowed(false));
			XMLConfiguration config = builder.getConfiguration();

			final String backColor = config.getString("colors.background");
			final String textColor = config.getString("colors.text");
			final String linkNormal = config.getString("colors.link[@normal]");
			final String defColor = config.getString("colors.default");

			final int rowsPerPage = config.getInt("rowsPerPage");

			final Object buttons = config.getList("buttons.name");
			System.out.println(buttons.getClass());

			System.out.println("colors.background: " + backColor);
			System.out.println("colors.text: " + textColor);
			System.out.println("colors.link[@normal]: " + linkNormal);
			System.out.println("colors.default: " + defColor);
			System.out.println("rowsPerPage: " + rowsPerPage);
			System.out.println("buttons.name: " + buttons);

			//
			config.setProperty("colors.foreground", "#000000");  // OK.
			//config.addProperty("colors.background", "#000000");
			//config.setReloadingStrategy(new FileChangedReloadingStrategy());

			builder.setAutoSave(true);
			//builder.save();
		}
		catch(ConfigurationException ex)
		{
			ex.printStackTrace();
		}
		finally
		{
		}
	}

}
