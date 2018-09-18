package tensorflow;

import org.tensorflow.*;
//import org.tensorflow.op.core.*;
import org.tensorflow.types.UInt8;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
//import java.io.File;
//import org.apache.commons.io.FileUtils;

public class TensorFlow_Main {

	// REF [site] >>
	//	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java
	//	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java/maven
	//	https://www.tensorflow.org/install/install_java

	public static void run(String[] args)
	{
		//runSimpleExample();

		//predictByInceptionGraph();

		// CNN model for MNIST.
		//	REF [file] >> mnist_cnn_tf.py & run_mnist_cnn.py in ${SWL_PYTHON_HOME}/test/machine_learning/tensorflow
		// TensorFlow checkpoint to TensorFlow SavedModel:
		//	REF [file] >> tensorflow_saving_and_loading.py in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/tensorflow
		predictByCnnSavedModel();
		//predictByCnnGraph();  // Not correctly working.
	}
	
	private static void runSimpleExample()
	{
		try (Graph graph = new Graph())
		{
			// Construct a graph to add two float tensors, using placeholders.
			//	z = x + y.
			Output x = graph.opBuilder("Placeholder", "x").setAttr("dtype", DataType.FLOAT).build().output(0);
			Output y = graph.opBuilder("Placeholder", "y").setAttr("dtype", DataType.FLOAT).build().output(0);
			Output z = graph.opBuilder("Add", "z").addInput(x).addInput(y).build().output(0);

			try (Session sess = new Session(graph))
			{
				// Execute the graph multiple times, each time with a different value of x and y.
				float[] X = new float[] {1, 2, 3};
				float[] Y = new float[] {4, 5, 6};
				for (int i = 0; i < X.length; ++i)
				{
					try (Tensor tx = Tensor.create(X[i]);
						Tensor ty = Tensor.create(Y[i]);
						Tensor tz = sess.runner().feed("x", tx).feed("y", ty).fetch("z").run().get(0))
					{
						System.out.println(X[i] + " + " + Y[i] + " = " + tz.floatValue());
					}
				}
			}
		}
	}
	
	// REF [file] >> ${TENSORFLOW_HOME}/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
	private static void predictByInceptionGraph()
	{
		// Save a graph in Python:
		//	tf.train.write_graph(session.graph_def, '/path/to/saved_dir, 'saved_graph.pb', as_text=False)
		//	tf.train.write_graph(session.graph_def, '/path/to/saved_dir, 'saved_graph.pbtxt', as_text=True)

		// Download Inception model:
		//	https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip

		final String modelDir = "data/machine_learning/tensorflow/inception5h";
		final String modelFile = "tensorflow_inception_graph.pb";
		final String labelFile = "imagenet_comp_graph_label_strings.txt";
		final String imageFilepath = "data/machine_learning/street2.jpg";
		
		final byte[] graphDef = readAllBytes(Paths.get(modelDir, modelFile));
		if (null == graphDef)
		{
			System.err.println("TensorFlow graph model not loaded.");
			return;
		}
		final List<String> labels = readAllLines(Paths.get(modelDir, labelFile));
		if (null == labels)
		{
			System.err.println("Labels not loaded.");
			return;
		}
		
		try (Graph graph = new Graph())
		{
			graph.importGraphDef(graphDef);
		    //graph.importGraphDef(FileUtils.readFileToByteArray(new File(Paths.get(modelDir, modelFile).toString())));

			try (Session sess = new Session(graph))
			{
				byte[] imageBytes = readAllBytes(Paths.get(imageFilepath));
				if (null == imageBytes)
				{
					System.err.println("Image not loaded.");
					return;
				}

				try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes))
				{
					try (Tensor<Float> pred = sess.runner().feed("input", image).fetch("output").run().get(0).expect(Float.class))
					{
						final long[] shape = pred.shape();
						if (2 != pred.numDimensions() || 1 != shape[0])
						{
							throw new RuntimeException(
								String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", Arrays.toString(shape))
							);
						}
	
						final int nlabels = (int)shape[1];
						final float[] labelProbabilities = pred.copyTo(new float[1][nlabels])[0];
						final int bestLabelIdx = getMaxIndex(labelProbabilities);
						System.out.println(
							String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f)
						);
					}
				}
			}
		}
	}

	// REF [site] >>
	//	https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example
	//		mnist_saved_model.py
	//	https://github.com/joyspark/TensorFlow
	//		simpleRegression.py & loadPythonModel.java
	private static void predictByCnnSavedModel()
	{
		// Save a TensorFlow SavedModel in Python:
		//	builder = tf.saved_model.builder.SavedModelBuilder('/path/to/saved_model')
		//	builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING], saver=saver)
		//	builder.save(as_text=False)

		final String[] imageFilepaths = {
			"data/machine_learning/mnist_img_1.jpg",  // 2.
			"data/machine_learning/mnist_img_2.jpg",  // 0.
			"data/machine_learning/mnist_img_3.jpg",  // 9.
			"data/machine_learning/mnist_img_4.jpg",  // 0.
			"data/machine_learning/mnist_img_5.jpg"  // 3.
		};

		// Load an image.
		byte[] imageBytes = readAllBytes(Paths.get(imageFilepaths[0]));
		if (null == imageBytes)
		{
			System.err.println("Image not loaded.");
			return;
		}

		// Create a saved model bundle.
		//try (SavedModelBundle b = SavedModelBundle.load("/path/to/saved_model", "serve"))
		try (SavedModelBundle b = SavedModelBundle.load("data/machine_learning/tensorflow/mnist_cnn_saved_model", "serve"))
		{
			// Create a session.
			Session sess = b.session();
			//Graph graph = b.graph();

			// Create an input tensor.
			try (Tensor<Float> image = constructTensorFromImage(imageBytes, 1))
			{
				/*
				// For training.
				float[][] outputLabels = new float[1][10];
				Tensor<Float> label = Tensor.create(outputLabels).expect(Float.class);
				//Operation op = graph.operation("output_tensor_ph");
				try (Tensor<Float> accuracy = sess.runner().feed("input_tensor_ph", image).feed("output_tensor_ph", label).fetch("accuracy/accuracy").run().get(0).expect(Float.class))
				*/
				// For prediction.
				try (Tensor<Float> pred = sess.runner().feed("input_tensor_ph", image).fetch("mnist_cnn_using_tf/fc2/fc/Softmax").run().get(0).expect(Float.class))
				{
					final long[] shape = pred.shape();
					if (2 != pred.numDimensions() || 1 != shape[0])
					{
						throw new RuntimeException(
							String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", Arrays.toString(shape))
						);
					}

					final int nlabels = (int)shape[1];
					final float[] labelProbabilities = pred.copyTo(new float[1][nlabels])[0];
					final int bestLabelIdx = getMaxIndex(labelProbabilities);
					System.out.println(
						String.format("BEST MATCH: %d (%.2f%% likely)", bestLabelIdx, labelProbabilities[bestLabelIdx] * 100f)
					);
				}
			}
		}
		catch (Exception ex)
		{
			System.err.println("Exception thrown: " + ex);
		}
	}

	private static void predictByCnnGraph()
	{
		// Save a graph in Python:
		//	tf.train.write_graph(session.graph_def, '/path/to/saved_dir, 'saved_graph.pb', as_text=False)
		//	tf.train.write_graph(session.graph_def, '/path/to/saved_dir, 'saved_graph.pbtxt', as_text=True)

		// NOTE [error] >> Attempting to use uninitialized value mnist_cnn_using_tf/fc1/fc/kernel.
		final String modelDir = "data/machine_learning/tensorflow";
		final String modelFile = "mnist_cnn_graph.pb";
		// NOTE [error] >> Invalid GraphDef.
		//final String modelDir = "data/machine_learning/tensorflow/mnist_cnn_saved_model";
		//final String modelFile = "saved_model.pb";
		final String[] imageFilepaths = {
			"data/machine_learning/mnist_img_1.jpg",  // 2.
			"data/machine_learning/mnist_img_2.jpg",  // 0.
			"data/machine_learning/mnist_img_3.jpg",  // 9.
			"data/machine_learning/mnist_img_4.jpg",  // 0.
			"data/machine_learning/mnist_img_5.jpg"  // 3.
		};

		// Load an image.
		byte[] imageBytes = readAllBytes(Paths.get(imageFilepaths[0]));
		if (null == imageBytes)
		{
			System.err.println("Image not loaded.");
			return;
		}
		
		final byte[] graphDef = readAllBytes(Paths.get(modelDir, modelFile));
		if (null == graphDef)
		{
			System.err.println("TensorFlow graph model not loaded.");
			return;
		}
		
		try (Graph graph = new Graph())
		{
			graph.importGraphDef(graphDef);
		    //graph.importGraphDef(FileUtils.readFileToByteArray(new File(Paths.get(modelDir, modelFile).toString())));

			try (Session sess = new Session(graph))
			{
				// Create an input tensor.
				try (Tensor<Float> image = constructTensorFromImage(imageBytes, 1))
				{
					/*
					// For training.
					float[][] outputLabels = new float[1][10];
					Tensor<Float> label = Tensor.create(outputLabels).expect(Float.class);
					//Operation op = graph.operation("output_tensor_ph");
					try (Tensor<Float> accuracy = sess.runner().feed("input_tensor_ph", image).feed("output_tensor_ph", label).fetch("accuracy/accuracy").run().get(0).expect(Float.class))
					*/
					// For prediction.
					try (Tensor<Float> pred = sess.runner().feed("input_tensor_ph", image).fetch("mnist_cnn_using_tf/fc2/fc/Softmax").run().get(0).expect(Float.class))
					{
						final long[] shape = pred.shape();
						if (2 != pred.numDimensions() || 1 != shape[0])
						{
							throw new RuntimeException(
								String.format("Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s", Arrays.toString(shape))
							);
						}
	
						final int nlabels = (int)shape[1];
						final float[] labelProbabilities = pred.copyTo(new float[1][nlabels])[0];
						final int bestLabelIdx = getMaxIndex(labelProbabilities);
						System.out.println(
							String.format("BEST MATCH: %d (%.2f%% likely)", bestLabelIdx, labelProbabilities[bestLabelIdx] * 100f)
						);
					}
				}
			}
		}
		catch (Exception ex)
		{
			System.err.println("Exception thrown: " + ex);
		}
	}

	private static Tensor<Float> constructTensorFromImage(byte[] imageBytes, long channels)
	{
		try (Graph g = new Graph())
		{
			GraphBuilder b = new GraphBuilder(g);

			// Since the graph is being constructed once per execution here, we can use a constant for the input image.
			// If the graph were to be re-used for multiple input images, a placeholder would have been more appropriate.
			final Output<String> input = b.constant("input", imageBytes);
			final Output<Float> output =
				b.expandDims(
					b.cast(b.decodeJpeg(input, channels), Float.class),
					b.constant("make_batch", 0)
				);
			try (Session sess = new Session(g))
			{
				// Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
				return sess.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		}
	}

	private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes)
	{
		try (Graph g = new Graph())
		{
			GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we can use a constant for the input image.
			// If the graph were to be re-used for multiple input images, a placeholder would have been more appropriate.
			final Output<String> input = b.constant("input", imageBytes);
			final Output<Float> output =
				b.div(
					b.sub(
						b.resizeBilinear(
							b.expandDims(
								b.cast(b.decodeJpeg(input, 3), Float.class),
								b.constant("make_batch", 0)
							),
							b.constant("size", new int[] {H, W})
						),
						b.constant("mean", mean)
					),
					b.constant("scale", scale)
				);
			try (Session sess = new Session(g))
			{
				// Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
				return sess.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		}
	}

	private static int getMaxIndex(float[] probabilities)
	{
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i)
		{
			if (probabilities[i] > probabilities[best])
			{
				best = i;
			}
		}
		return best;
	}

	private static byte[] readAllBytes(Path path)
	{
		try
		{
			return Files.readAllBytes(path);
		}
		catch (IOException ex)
		{
			//System.err.println("Failed to read [" + path + "]: " + ex.getMessage());
			//System.exit(1);
		}
		return null;
	}

	private static List<String> readAllLines(Path path)
	{
		try
		{
			return Files.readAllLines(path, Charset.forName("UTF-8"));
		}
		catch (IOException ex)
		{
			//System.err.println("Failed to read [" + path + "]: " + ex.getMessage());
			//System.exit(0);
		}
		return null;
	}

	// In the fullness of time, equivalents of the methods of this class should be auto-generated from the OpDefs linked into libtensorflow_jni.so.
	// That would match what is done in other languages like Python, C++ and Go.
	static class GraphBuilder
	{
		GraphBuilder(Graph g)
		{
			this.g = g;
		}

		<T> Output<T> add(Output<T> x, Output<T> y)
		{
			return binaryOp("Add", x, y);
		}

		<T> Output<T> sub(Output<T> x, Output<T> y)
		{
			return binaryOp("Sub", x, y);
		}

		Output<Float> mul(Output<Float> x, Output<Float> y)
		{
			return binaryOp("Mul", x, y);
		}

		Output<Float> div(Output<Float> x, Output<Float> y)
		{
			return binaryOp("Div", x, y);
		}

		<T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size)
		{
			return binaryOp3("ResizeBilinear", images, size);
		}

		<T> Output<T> expandDims(Output<T> input, Output<Integer> dim)
		{
			return binaryOp3("ExpandDims", input, dim);
		}

		<T, U> Output<U> cast(Output<T> value, Class<U> type)
		{
			DataType dtype = DataType.fromClass(type);
			return g.opBuilder("Cast", "Cast")
				.addInput(value)
				.setAttr("DstT", dtype)
				.build()
				.<U>output(0);
		}

		Output<UInt8> decodeJpeg(Output<String> contents, long channels)
		{
			return g.opBuilder("DecodeJpeg", "DecodeJpeg")
				.addInput(contents)
				.setAttr("channels", channels)
				.build()
				.<UInt8>output(0);
		}

		<T> Output<T> placeholder(String name, Class<T> type)
		{
			return g.opBuilder("Placeholder", name)
				.setAttr("dtype", DataType.fromClass(type))
				.build()
				.<T>output(0);
		}

		<T> Output<T> constant(String name, Object value, Class<T> type)
		{
			try (Tensor<T> t = Tensor.<T>create(value, type))
			{
				return g.opBuilder("Const", name)
					.setAttr("dtype", DataType.fromClass(type))
					.setAttr("value", t)
					.build()
					.<T>output(0);
			}
		}

		Output<String> constant(String name, byte[] value)
		{
			return this.constant(name, value, String.class);
		}

		Output<String> constant(String name, byte[][] value)
		{
			return this.constant(name, value, String.class);
		}

		Output<Integer> constant(String name, int value)
		{
			return this.constant(name, value, Integer.class);
		}

		Output<Integer> constant(String name, int[] value)
		{
			return this.constant(name, value, Integer.class);
		}

		Output<Float> constant(String name, float value)
		{
			return this.constant(name, value, Float.class);
		}

		private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2)
		{
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2)
		{
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private Graph g;
	}
}
