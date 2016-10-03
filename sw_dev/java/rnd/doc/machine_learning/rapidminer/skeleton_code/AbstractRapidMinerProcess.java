package ml.rapidminer;

import java.io.File;
import java.io.IOException;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import com.rapidminer.Process;
import com.rapidminer.RapidMiner;
import com.rapidminer.RapidMiner.ExecutionMode;
import com.rapidminer.tools.XMLException;

/**
 * RapidMiner의 프로세스를 수행하기 위해 공통 속성과 기능을 정의한 추상 클래스
 */
public abstract class AbstractRapidMinerProcess
										implements Runnable {
	
	private static 	boolean			initRapidminer = true;
	protected 	Process				proc;						// RapidMiner 프로세스
	protected	File				procConfigFile;				// 프로세스 내용이 정의된 파일
	
	public static void init() {
		if(initRapidminer) {
			RapidMiner.setExecutionMode(ExecutionMode.EMBEDDED_WITHOUT_UI);
			System.setProperty("rapidminer.home", new File("res/rapidminer/home").getAbsolutePath());
			
			// 패러랠 플러그인 활성화
			com.rapidminer.ParallelPluginInit.initPlugin();
			com.rapidminer.PluginInitWekaExtension.initPluginManager();
			
			RapidMiner.init();
			initRapidminer = false;
		}
	}
	
	/**
	 * RapidMiner 프로세스를 수행하는 클래스를 생성한다
	 * 
	 * @param processConfig			프로세스 내용이 정의된 파일
	 */
	public AbstractRapidMinerProcess(String processConfig) {
		Logger logger = LogManager.getLogger(AbstractRapidMinerProcess.class);
		procConfigFile = new File(processConfig);
		
		if(initRapidminer) {
			AbstractRapidMinerProcess.init();
		}
	
		if(procConfigFile.exists()) {
			try {
				proc = new Process(procConfigFile);
			} catch (Exception e) {
				logger.error(e);
			}
		}
		else {
			logger.fatal("RapidMiner Process file " + procConfigFile.getAbsolutePath() + " is not exist!!!");
			System.exit(-1);
		}
	}
	
	// Getter & Setter
	
	/**
	 * 프로세스 파일 경로를 얻어낸다
	 * @return
	 */
	public String getProcessConfigureationFile() {
		if(procConfigFile.exists()) { return procConfigFile.getAbsolutePath(); }
		return null;
	}
	
	/**
	 * 프로세스 파일을 설정한다
	 * @param procConfigFile
	 */
	public void setProcessConfigureation(String processConfig) {
		procConfigFile = new File(processConfig);
		try {
			proc = new Process(procConfigFile);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (XMLException e) {
			e.printStackTrace();
		}
	}
}
