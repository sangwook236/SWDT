package configuration;

import org.apache.commons.configuration2.*;
import org.apache.commons.configuration2.ex.*;
import org.apache.commons.configuration2.ex.ConfigurationException;
import org.apache.commons.configuration2.builder.FileBasedConfigurationBuilder;
import org.apache.commons.configuration2.builder.fluent.Parameters;
import org.apache.commons.configuration2.convert.DefaultListDelimiterHandler;

import java.awt.Dimension;
//import java.util.ArrayList;

final class PropertiesConfigurationTest {

	public static void run(String[] args)
	{
		try
		{
			FileBasedConfigurationBuilder<PropertiesConfiguration> builder =
				    new FileBasedConfigurationBuilder<PropertiesConfiguration>(PropertiesConfiguration.class)
				    .configure(new Parameters().properties()
				        .setFileName("src/configuration/user_gui.properties")
				        .setThrowExceptionOnMissing(true)
				        .setListDelimiterHandler(new DefaultListDelimiterHandler(';'))
				        .setIncludesAllowed(false));
			Configuration config = builder.getConfiguration();

			final String foreColor = config.getString("colors.foreground");
			final String backColor = config.getString("colors.background");
			final Dimension size = new Dimension(config.getInt("window.width"), config.getInt("window.height"));

			final String[] colors = config.getStringArray("colors.pie");
			final Object colorList = config.getList("colors.pie");
			System.out.println(colorList.getClass());

			System.out.println("colors.foreground: " + foreColor);
			System.out.println("colors.background: " + backColor);
			System.out.println("window.width: " + size.getWidth() + ", window.height:" + size.getHeight());

			System.out.print("colors.pie (string): ");
			for (int i = 0; i < colors.length; ++i)
				System.out.print(colors[i] + ", ");
			System.out.println();
			System.out.println("colors.pie (list): " + colorList);

			//
			config.setProperty("colors.background", "#000000");
			//((PropertiesConfiguration)config).setReloadingStrategy(new FileChangedReloadingStrategy());

			builder.setAutoSave(true);
			builder.save();
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
