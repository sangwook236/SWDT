package weka;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;

class Example1 {
	public static void run(String[] args)
	{
		try
		{
			Instances dataset = DataSource.read("./data/machine_learning/weka/iris.arff");

			if (dataset.classIndex() == -1)
				dataset.setClassIndex(dataset.numAttributes() - 1);

			//
			String[] options = new String[1];
			options[0] = "-C 0.25 -M 2";

			J48 model = new J48();
			model.setOptions(options);

			model.buildClassifier(dataset);

			//
			Evaluation eval = new Evaluation(dataset);

			eval.crossValidateModel(model, dataset, 10, new Random(10));
			System.out.println(eval.toSummaryString("\nResults\n\n", false));

			//
			{
				DataSink.write("./data/machine_learning/weka/example1.xrff", dataset);
				//DataSink.write("./data/machine_learning/weka/example1.csv", dataset);

				CSVSaver saver = new CSVSaver();
				saver.setInstances(dataset);
				saver.setFile(new java.io.File("./data/machine_learning/weka/example1.csv"));
				saver.writeBatch();
			}

			//

			// Use Java serialized object.
			try
			{
				String modelFileName = "./data/machine_learning/weka/example1.model";

				//
				java.io.ObjectOutputStream ostream = new java.io.ObjectOutputStream(new java.io.FileOutputStream(modelFileName));
				ostream.writeObject(model);
				ostream.close();

				ostream = null;

				//
				java.io.ObjectInputStream istream = new java.io.ObjectInputStream(new java.io.FileInputStream(modelFileName));

				J48 readModel = (J48)istream.readObject();

				istream.close();
				istream = null;

				//
				options[0] = "-C 0.5 -M 4";

				readModel.setOptions(options);
				readModel.buildClassifier(dataset);
			}
			catch (java.io.IOException eX)
			{
				System.out.println("I/O exception occurred: " + eX.getMessage());
			}
		}
		catch (Exception eX)
		{
			eX.printStackTrace();
		}
	}
}
