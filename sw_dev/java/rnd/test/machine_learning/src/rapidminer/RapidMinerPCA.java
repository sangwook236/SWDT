package rapidminer;

//import java.util.ArrayList;
//import java.util.List;
import java.io.File;
import java.io.IOException;

//import com.rapidminer.example.Attribute;
import com.rapidminer.example.ExampleSet;
//import com.rapidminer.example.table.AttributeFactory;
//import com.rapidminer.example.table.DoubleArrayDataRow;
//import com.rapidminer.example.table.MemoryExampleTable;
import com.rapidminer.gui.renderer.models.EigenvectorModelEigenvalueRenderer.EigenvalueTableModel;
import com.rapidminer.gui.renderer.models.EigenvectorModelEigenvectorRenderer.EigenvectorTableModel;
import com.rapidminer.operator.IOContainer;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.features.transformation.PCAModel;
//import com.rapidminer.tools.Ontology;


/**
 * RapidMiner의 PCA 과정을 래핑한 래퍼 클래스.
 */
public class RapidMinerPCA extends AbstractRapidMinerProcess
{
	/**
	 * 디폴트 프로세스로 PCA를 수행하는 컴포넌트를 생성한다.
	 */
	public RapidMinerPCA()
	{
		super("data/machine_learning/rapidminer/proc_PCA.xml");
	}

	/**
	 * PCA를 수행하는 컴포넌트를 생성한다.
	 * 
	 * @param processFile PCA 프로세스가 저장된 RapidMiner 파일 경로.
	 */
	public RapidMinerPCA(String processFile)
	{
		super(processFile);
	}

	/**
	 */
	@Override
	public void run()
	{
		if (null == example_)
		{
			System.err.println("ExampleSet is null - loadData() before run().");
			return;
		}

		// 프로세스.
		try
		{
			// 미리 설정되어 있는 RapidMiner 과정을 수행한다.
			IOContainer c = proc_.run(new IOContainer(example_));

			// PCA Model 을 얻고 고유벡터와 고유값을 다른 형태로 저장한다.
			//com.rapidminer.example.set.SimpleExampleSet exampleSet = (com.rapidminer.example.set.SimpleExampleSet)c.getElementAt(0);
			//com.rapidminer.example.set.SimpleExampleSet original = (com.rapidminer.example.set.SimpleExampleSet)c.getElementAt(1);
			PCAModel pcaModel = (PCAModel)c.getElementAt(2);

			// 고유값.
			EigenvalueTableModel eigenValMat = pcaModel.getEigenvalueTableModel();			
			// 고유벡터.
			EigenvectorTableModel eigenVecMat = pcaModel.getEigenvectorTableModel();

			System.out.println("Eigen Values : " + eigenValMat.getRowCount() + " rows.");
			System.out.println("Eigen Vectors : " + eigenVecMat.getColumnCount() + " x " + eigenVecMat.getRowCount() + " matrix.");
		}
		catch (Exception ex)
		{
			ex.printStackTrace();
		}
	}

	/**
	 */
	public void loadData()
	{
		String filepath = "./data/machine_learning/rapidminer/iris.csv";
		com.rapidminer.tools.LineParser parser = new com.rapidminer.tools.LineParser();
		try
		{
			parser.setSplitExpression(com.rapidminer.tools.LineParser.SPLIT_BY_COMMA_EXPRESSION);
		}
		catch (OperatorException ex)
		{
			// TODO Auto-generated catch block
			ex.printStackTrace();
		}
		com.rapidminer.gui.tools.dialogs.wizards.dataimport.csv.CSVFileReader reader = new com.rapidminer.gui.tools.dialogs.wizards.dataimport.csv.CSVFileReader(new File(filepath), true, parser, java.text.NumberFormat.getInstance()); 

		try
		{
			example_ = reader.createExampleSet();
			com.rapidminer.example.Attributes attributes = example_.getAttributes();
			attributes.setLabel(attributes.get("label"));
			attributes.remove(attributes.get("id"));
		}
		catch (IOException ex)
		{
			// TODO Auto-generated catch block
			ex.printStackTrace();
		}
	}

	// RapidMiner 데이터.
	private ExampleSet example_ = null;
}
