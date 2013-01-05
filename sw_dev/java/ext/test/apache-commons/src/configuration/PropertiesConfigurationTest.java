package configuration;

import org.apache.commons.configuration.*;
import org.apache.commons.configuration.reloading.*;

import java.awt.Dimension;
//import java.util.ArrayList;

final class PropertiesConfigurationTest {
	public static void run(String[] args)
	{
		try
		{
			Configuration config = new PropertiesConfiguration("src/configuration/user_gui.properties");
			
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
			((PropertiesConfiguration)config).setReloadingStrategy(new FileChangedReloadingStrategy());
			
			((AbstractFileConfiguration)config).setAutoSave(true);
			//((PropertiesConfiguration)config).save();
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
