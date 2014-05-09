package rapidminer;

import java.io.File;
import java.io.IOException;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import com.rapidminer.Process;
import com.rapidminer.RapidMiner;
import com.rapidminer.RapidMiner.ExecutionMode;
import com.rapidminer.tools.XMLException;

/**
 * RapidMiner의 프로세스를 수행하기 위해 공통 속성과 기능을 정의한 추상 클래스.
 */
public abstract class AbstractRapidMinerProcess implements Runnable
{
	public static void init()
	{
		if (initRapidminer_)
		{
			RapidMiner.setExecutionMode(ExecutionMode.EMBEDDED_WITHOUT_UI);
			System.setProperty("rapidminer.home", new File("D:/MyProgramFiles/Rapid-I/RapidMiner5").getAbsolutePath());
			
			// activate parallel plug-in.
			//com.rapidminer.ParallelPluginInit.initPlugin();
			//com.rapidminer.PluginInitWekaExtension.initPluginManager();
			
			RapidMiner.init();
			initRapidminer_ = false;
		}
	}
	
	/**
	 * RapidMiner 프로세스를 수행하는 클래스를 생성한다.
	 * 
	 * @param processConfig 프로세스 내용이 정의된 파일.
	 */
	public AbstractRapidMinerProcess(String processConfig)
	{
		Logger logger = LogManager.getLogger(AbstractRapidMinerProcess.class);
		procConfigFile_ = new File(processConfig);
		
		if (initRapidminer_)
		{
			AbstractRapidMinerProcess.init();
		}
	
		if (procConfigFile_.exists())
		{
			try
			{
				proc_ = new Process(procConfigFile_);
			}
			catch (Exception e)
			{
				logger.error(e);
			}
		}
		else
		{
			logger.fatal("RapidMiner Process file " + procConfigFile_.getAbsolutePath() + " is not exist!!!");
			System.exit(-1);
		}
	}
	
	// Getter & Setter
	
	/**
	 * 프로세스 파일 경로를 얻어낸다
	 * @return
	 */
	public String getProcessConfigureationFile()
	{
		return procConfigFile_.exists() ? procConfigFile_.getAbsolutePath() : null;
	}
	
	/**
	 * 프로세스 파일을 설정한다
	 * @param procConfigFile_
	 */
	public void setProcessConfigureation(String processConfig)
	{
		procConfigFile_ = new File(processConfig);
		try
		{
			proc_ = new Process(procConfigFile_);
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
		catch (XMLException e)
		{
			e.printStackTrace();
		}
	}

	private static boolean initRapidminer_ = true;
	protected Process proc_;  // RapidMiner process.
	protected File procConfigFile_;  // a file in which a process is defined 
}
