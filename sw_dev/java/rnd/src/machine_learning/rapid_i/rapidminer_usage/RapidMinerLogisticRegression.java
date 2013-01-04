package ml.rapidminer;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import com.rapidminer.datatable.DataTable;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.AttributeFactory;
import com.rapidminer.example.table.DataRowFactory;
import com.rapidminer.example.table.MemoryExampleTable;
import com.rapidminer.operator.IOContainer;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.learner.functions.kernel.JMySVMModel;
import com.rapidminer.operator.learner.functions.kernel.MyKLRModel;
import com.rapidminer.tools.Ontology;

/**
 *	Logistic Regression을 수행하는 RapidMiner 프로세스
 */
public class RapidMinerLogisticRegression extends AbstractRapidMinerProcess {
	
	static Logger		log = Logger.getLogger(RapidMinerLogisticRegression.class);
	private double		bias = 0;
	private double		weights[];
	private boolean		svmMode = false;
	
	private	 ExampleSet example = null;
	
	public RapidMinerLogisticRegression() {
		this(false);
	}
	
	public RapidMinerLogisticRegression(boolean svmMode) {
		this("res/rapidminer/proc_logistic_regression.xml");
		this.svmMode = svmMode;
		if(svmMode) {
			setProcessConfigureation("res/rapidminer/proc_svm.xml");
		}
	}

	public RapidMinerLogisticRegression(String processConfig) {
		super(processConfig);		
	}
	
	/**
	 * 트레이닝 데이터를 설정한다.
	 */
	public void setData(List<Object> positive, List<Object> negative){
		
		double[] row = (double[]) positive.get(0);
		int colNum = row.length;
		
		// Attribute 목록을 구성, 마지막 컬럼은 레이블이라는 것을 미리 알고 있음
		List<Attribute> attr = new ArrayList<Attribute>();
		Attribute[] attrArray = new Attribute[colNum + 1];
		for(int i = 0; i < colNum; i++) {
			Attribute a = AttributeFactory.createAttribute("x" + i,  Ontology.NUMERICAL);
			attr.add(a);
			attrArray[i] = a;
		}
		Attribute label = AttributeFactory.createAttribute("label", Ontology.BINOMINAL);
		attr.add(label);
		attrArray[colNum] = label;
		
		// ExampleSet을 생성
		MemoryExampleTable table = new MemoryExampleTable(attr);
		DataRowFactory df = new DataRowFactory(DataRowFactory.TYPE_DOUBLE_ARRAY, '.');
		
		Object[] exampleData;
		
		// Positive
		for(int i = 0; i < positive.size(); i++) {
			exampleData = new Object[colNum + 1];
			row = (double[]) positive.get(i);
			for(int ci = 0; ci < row.length; ci++) {
				exampleData[ci] = row[ci];
			}
			exampleData[row.length] = "Positive";
			table.addDataRow(df.create(exampleData, attrArray));
		}
		
		// Negative
		for(int i = 0; i < negative.size(); i++) {
			exampleData = new Object[colNum + 1];
			row = (double[]) negative.get(i);			
			for(int ci = 0; ci < row.length; ci++) {
				exampleData[ci] = row[ci];
			}
			exampleData[row.length] = "Negative";
			table.addDataRow(df.create(exampleData, attrArray));
		}
		
		example = table.createExampleSet(label);
	}

	@Override
	public void run() {
		try {
			IOContainer input = new IOContainer(example);
			IOContainer ret = proc.run(input);
			
			/*
			for(int i = 0; i < ret.getIOObjects().length; i++) {
				IOObject obj = ret.getElementAt(i);
				System.out.println("========================================================");
				System.out.println("[" + i + "] " +  obj.getClass());
				System.out.println("========================================================");
				System.out.println(obj);
			}
			*/
			
			if(!svmMode) {
				MyKLRModel krlModel = (MyKLRModel) ret.getElementAt(1);
				DataTable wt = krlModel.createWeightsTable();
				
				bias = krlModel.getBias();
				weights = new double[wt.getRowNumber()];
				for(int i = 0; i < wt.getRowNumber(); i++) {
					weights[i] = wt.getRow(i).getValue(1);
				}
			} else {
				JMySVMModel svmModel = (JMySVMModel) ret.getElementAt(1);
				DataTable wt = svmModel.createWeightsTable();
				
				bias = svmModel.getBias();
				weights = new double[wt.getRowNumber()];
				for(int i = 0; i < wt.getRowNumber(); i++) {
					weights[i] = wt.getRow(i).getValue(1);
				}
			}
		} catch (OperatorException e) {
			e.printStackTrace();
		}
	}
	
	public double getBias() {
		return bias;
	}
	
	public double[] getWeights() {
		return weights;
	}
}
