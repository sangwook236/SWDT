package com.sangwook.tensorflowapp;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.annotation.Nullable;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class TensorFlowInterfaceForFrozenGraph {
	public TensorFlowInterfaceForFrozenGraph(AssetManager assetManager)
	{
		inferrer_ = new TensorFlowInferenceInterface(assetManager, model_);
	}

	public float[] infer(final byte[] input)
	{
		// Preprocessing.
		final float[] inputFloat = new float[input.length];
		for (int i = 0; i < input.length; ++i) {
			//inputFloat[i] = (float)Byte.toUnsignedInt(input[i]);
			inputFloat[i] = (float)Byte.toUnsignedInt(input[i]) / 255.0f;
		}

		// Copy the input data into TensorFlow.
		inferrer_.feed(inputNodeName_, inputFloat, inputShape_);

		// Run inference.
		final String[] outputNodeNames = {outputNodeName_};
		inferrer_.run(outputNodeNames);

		// Copy the output Tensor back into the output array.
		float[] outputs = new float[10];
		inferrer_.fetch(outputNodeName_, outputs);

		return outputs;
	}

	//private final String model_ = "file:///android_asset/mnist_cnn_frozen_graph.pb";
	private final String model_ = "file:///android_asset/mnist_cnn_optimized_frozen_graph.pb";
	private final long[] inputShape_ = {1, 28, 28, 1};
	private final String inputNodeName_ = "input_tensor_ph";
	private final String outputNodeName_ = "mnist_cnn_using_tf/fc2/fc/Softmax";

	private TensorFlowInferenceInterface inferrer_;
}
