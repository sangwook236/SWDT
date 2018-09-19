package com.sangwook.tensorflowapp;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.*;
//import org.tensorflow.op.core.*;
import org.tensorflow.types.UInt8;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
//import java.io.File;
//import org.apache.commons.io.FileUtils;

public class TensorFlowInterfaceForSavedModel {

	public static int predictByCnnServingModel(String modelDirPath, byte[] imageBytes)
	{
		Log.i("[SWL-Mobile]", "**************************** 1");
		Log.i("[SWL-Mobile]", "TensorFlow version: " + TensorFlow.version());
		Log.i("[SWL-Mobile]", "**************************** 2");

		// Create a saved model bundle.
		try (SavedModelBundle b = SavedModelBundle.load(modelDirPath, "serve"))
		{
			Log.i("[SWL-Mobile]", "**************************** 3");

			// Create a session.
			Session sess = b.session();
			//Graph graph = b.graph();

			// Create an input tensor.
			try (Tensor<Float> image = constructTensorFromImage(imageBytes, 1))
			{
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
					Log.i("[SWL-Mobile]", String.format("BEST MATCH: %d (%.2f%% likely)", bestLabelIdx, labelProbabilities[bestLabelIdx] * 100f));

					return bestLabelIdx;
				}
			}
		}
		catch (Exception ex)
		{
			Log.e("[SWL-Mobile]", "Exception thrown: " + ex);
		}

		return -1;
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
			//return Files.readAllBytes(path);
			Bitmap bitmap = BitmapFactory.decodeFile(path.toString());
			ByteArrayOutputStream stream = new ByteArrayOutputStream();
			bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
			byte[] imageBytes = stream.toByteArray();
			stream.close();
			return imageBytes;
		}
		catch (IOException ex)
		{
			//System.err.println("Failed to read [" + path + "]: " + ex.getMessage());
			//System.exit(1);
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
