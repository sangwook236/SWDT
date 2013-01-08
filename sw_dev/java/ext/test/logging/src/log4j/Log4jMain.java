package log4j;

import org.apache.log4j.Logger;
import org.apache.log4j.BasicConfigurator;

public final class Log4jMain {

	/**
	 * @param args
	 */
	public static void run(String[] args)
	{
		// Set up a simple configuration that logs on the console.
		BasicConfigurator.configure();

		logger.info("Entering application.");
		Bar bar = new Bar();
		bar.doIt();
		logger.info("Exiting application.");
	}

	static Logger logger = Logger.getLogger(Log4jMain.class);
}

class Bar {
	public void doIt()
	{
		logger.debug("Did it again!");
	}

	static Logger logger = Logger.getLogger(Bar.class);
}
