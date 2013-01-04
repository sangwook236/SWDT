package ml.rapidminer;

import java.util.ArrayList;
import java.util.List;

import ml.PrincipalComponentAnalysis;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import bin.util.opencv.MatUtil;

import com.googlecode.javacv.cpp.opencv_core.CvMat;
import static com.googlecode.javacv.cpp.opencv_core.*;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.AttributeFactory;
import com.rapidminer.example.table.DoubleArrayDataRow;
import com.rapidminer.example.table.MemoryExampleTable;
import com.rapidminer.gui.renderer.models.EigenvectorModelEigenvalueRenderer.EigenvalueTableModel;
import com.rapidminer.gui.renderer.models.EigenvectorModelEigenvectorRenderer.EigenvectorTableModel;
import com.rapidminer.operator.IOContainer;
import com.rapidminer.operator.features.transformation.PCAModel;
import com.rapidminer.tools.Ontology;


/**
 * RapidMiner의 PCA 과정을 래핑한 래퍼 클래스
 */
public class RapidMinerPCA extends AbstractRapidMinerProcess implements PrincipalComponentAnalysis {
	
	private		Logger							log = LogManager.getLogger(RapidMinerPCA.class);	
	
	// RapidMiner 프로세스
	private 	CvMat							eigenValue = null;
	private 	CvMat							eigenVector = null;
	
	// RapidMiner 데이터
	private		ExampleSet					example = null;
	
	/**
	 * 디폴트 프로세스로 PCA를 수행하는 컴포넌트를 생성한다
	 */
	public RapidMinerPCA() {
		super("res/rapidminer/proc_pca.xml");
	}
	
	/**
	 * PCA를 수행하는 컴포넌트를 생성한다
	 * 
	 * @param processFile				PCA 프로세스가 저장된 RapidMiner 파일 경로
	 */
	public RapidMinerPCA(String processFile) {
		super(processFile);
	}
	
	/* (non-Javadoc)
	 * @see ml.rapidminer.PrincipalComponentAnalysis#run()
	 */
	@Override
	public void run() {
		
		// Pre-Condition 체크
		if(example == null) {
			log.error("ExampleSet is null - setData() before run()");
			return;
		}
		
		// 프로세스
		try {
			// 미리 설정되어 있는 RapidMiner 과정을 수행한다
			IOContainer c = proc.run(new IOContainer(example));
			
			// PCA Model 을 얻고 고유벡터와 고유값을 다른 형태로 저장한다 
			PCAModel pcaModel = (PCAModel) c.getElementAt(1);
			
			// 고유값
			EigenvalueTableModel eigenValMat = pcaModel.getEigenvalueTableModel();			
			eigenValue = MatUtil.createMat(eigenValMat.getRowCount(), eigenValMat.getColumnCount() - 1, CV_32FC1);
			for(int i = 0; i < eigenValMat.getRowCount(); i++) {
				for(int j = 1; j < eigenValMat.getColumnCount(); j++) {
					try {
						eigenValue.put(i, j - 1, Float.parseFloat(eigenValMat.getValueAt(i, j).toString()));
					} catch (NumberFormatException nfe) {
						log.warn("EigenValue Number Format Exception at (" + i + ", " + j + ") : " + eigenValMat.getValueAt(i, j));
					}
				}
			}
			// 고유벡터
			EigenvectorTableModel eigenVecMat = pcaModel.getEigenvectorTableModel();
			eigenVector = MatUtil.createMat(eigenVecMat.getRowCount(), eigenVecMat.getColumnCount() - 1, CV_32FC1);
			for(int i = 0; i < eigenVecMat.getRowCount(); i++) {
				for(int j = 1; j < eigenVecMat.getColumnCount(); j++) {
					try {
						eigenVector.put(i, j - 1, Float.parseFloat(eigenVecMat.getValueAt(i, j).toString()));
					} catch (NumberFormatException nfe) {
						eigenVector.put(i, j - 1, 0);
						log.warn("EigenVector Number Format Exception at (" + i + ", " + j + ") : " + eigenVecMat.getValueAt(i, j));
					}
				}
			}
			
			log.info("Eigen Values : " + eigenValue.rows()+ " rows");
			log.info("Eigen Vectors : " + eigenVector.cols() + " x " + eigenVector.rows()+ " matrix");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/* (non-Javadoc)
	 * @see ml.rapidminer.PrincipalComponentAnalysis#setData(com.googlecode.javacv.cpp.opencv_core.CvMat)
	 */
	@Override
	public void setData(CvMat data) {
		// 매개변수 체크
		if(data == null) {
			log.error("setData() - Argument is null");
			return;
		}
		if(data.cols() <= 0 || data.rows() <= 0){
			log.error("setData() - Invalid argument cols = " + data.cols() + ", rows = " + data.rows());
			return;
		}
		
		// Attribute 목록을 구성
		List<Attribute> attr = new ArrayList<Attribute>();
		for(int i = 0; i < data.cols(); i++) {
			attr.add(AttributeFactory.createAttribute("x" + i,  Ontology.NUMERICAL));
		}
		
		// ExampleSet을 생성
		MemoryExampleTable table = new MemoryExampleTable(attr);		
		for(int i = 0; i < data.rows(); i++) {						
			// Row 데이터를 채워넣음
			double[] dataRow = new double[data.cols()];
			for(int j = 0; j < data.cols(); j++) {
				dataRow[j] = data.get(i, j);
			}
			table.addDataRow(new DoubleArrayDataRow(dataRow));
		}
		example = table.createExampleSet();
	}
	
	/* (non-Javadoc)
	 * @see ml.rapidminer.PrincipalComponentAnalysis#getEigenValue()
	 */
	@Override
	public CvMat getEigenValue() {
		return eigenValue;
	}
	
	/* (non-Javadoc)
	 * @see ml.rapidminer.PrincipalComponentAnalysis#getEigenVector()
	 */
	@Override
	public CvMat getEigenVector() {
		return eigenVector;
	}
}
