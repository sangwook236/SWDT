package log4j;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.*;

public final class Log4j_Main {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		// Set up a simple configuration that logs on the console.
		Configurator.initialize(new DefaultConfiguration());

		logger.info("Entering application.");
		Bar bar = new Bar();
		bar.doIt();
		logger.info("Exiting application.");
	}

	static Logger logger = LogManager.getLogger(Log4j_Main.class);
}

class Bar {
	public void doIt()
	{
		logger.debug("Did it again!");
	}

	static Logger logger = LogManager.getLogger(Bar.class);
}
