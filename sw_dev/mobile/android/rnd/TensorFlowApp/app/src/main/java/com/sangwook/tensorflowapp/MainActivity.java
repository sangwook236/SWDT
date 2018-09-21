package com.sangwook.tensorflowapp;

import android.Manifest;
import android.os.Environment;
import android.os.Bundle;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.TextView;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Properties;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Properties p = System.getProperties();
        System.setProperty("org.tensorflow.NativeLibrary.DEBUG", "1");
        //Log.i("[SWL-Mobile]", "" + System.getProperties().toString());

        // Request the read/write permissions.
        MainActivity.requestStoragePermissions(this);

        //
        if (isExternalStorageReadable())
            Log.i("[SWL-Mobile]", "External storage readable.");
        else
            Log.w("[SWL-Mobile]", "External storage NOT readable.");

        //final String rootDirPath = Environment.getRootDirectory().getAbsolutePath();
        //final String dataDirPath = Environment.getDataDirectory().getAbsolutePath();
        final String extStorageDirPath = Environment.getExternalStorageDirectory().getAbsolutePath();

        final String[] imageFilepaths = {
                extStorageDirPath + "/mnist_img_2.raw",  // 0.
                extStorageDirPath + "/mnist_img_23.raw",  // 1.
                extStorageDirPath + "/mnist_img_1.raw",  // 2.
                extStorageDirPath + "/mnist_img_5.raw",  // 3.
                extStorageDirPath + "/mnist_img_26.raw",  // 4.
                extStorageDirPath + "/mnist_img_11.raw",  // 5.
                extStorageDirPath + "/mnist_img_35.raw",  // 6.
                extStorageDirPath + "/mnist_img_6.raw",  // 7.
                extStorageDirPath + "/mnist_img_55.raw",  // 8.
                extStorageDirPath + "/mnist_img_3.raw"  // 9.
        };
        final String imageFilepath = imageFilepaths[0];

        // Load an image.
        final byte[] imageBytes = readAllBytes(Paths.get(imageFilepath));
        if (null == imageBytes)
        {
            Log.e("[SWL-Mobile]", "Image not loaded: " + imageFilepath);
            return;
        }

        final int bestLabelIdx = predictByFrozenGraph(imageBytes);
        //final int bestLabelIdx = predictBySavedModel(imageBytes, extStorageDirPath);  // Not working.

        // Example of a call to a native method
        TextView tv = (TextView)findViewById(R.id.sample_text);
        //tv.setText(stringFromJNI());
        tv.setText("Predicted class = " + bestLabelIdx);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    private int predictByFrozenGraph(byte[] imageBytes)
    {
        TensorFlowInterfaceForFrozenGraph tensorFlowInterface = new TensorFlowInterfaceForFrozenGraph(getAssets());
        final float[] outputs = tensorFlowInterface.infer(imageBytes);
        Log.i("[SWL-Mobile]", "***** Output probabilities = " + Arrays.toString(outputs));

        // Find the best classifications.
        ArrayList<Float> outputList = new ArrayList<Float>(outputs.length);
        for (float val : outputs)
            outputList.add(val);
        final int bestLabelIdx = outputList.indexOf(Collections.max(outputList));
        Log.i("[SWL-Mobile]", "***** Best label index = " + bestLabelIdx);

        return bestLabelIdx;
    }

    private int predictBySavedModel(byte[] imageBytes, String extStorageDirPath)
    {
        final String modelDirPath = extStorageDirPath + "/mnist_cnn_saved_model";

        final int bestLabelIdx = TensorFlowInterfaceForSavedModel.predictByCnnServingModel(modelDirPath, imageBytes);
        Log.i("[SWL-Mobile]", "***** Best label index = " + bestLabelIdx);

        return bestLabelIdx;
    }

    @Nullable
    public static byte[] readAllBytes(Path path)
    {
        try
        {
            return Files.readAllBytes(path);
			/*
			// FIXME [error] >> Do not read bytes of actual image size.
			Bitmap bitmap = BitmapFactory.decodeFile(path.toString());
			ByteArrayOutputStream stream = new ByteArrayOutputStream();
			bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
			final byte[] image = stream.toByteArray();
			stream.close();
			return image;
			*/
        }
        catch (IOException ex)
        {
            Log.e("[SWL-Mobile]", "Exception thrown: Failed to read '" + path + "' " + ex);
        }
        return null;
    }

    boolean isExternalStorageWritable()
    {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state);
    }

    boolean isExternalStorageReadable()
    {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state) || Environment.MEDIA_MOUNTED_READ_ONLY.equals(state);
    }

    // NOTE [info] >> Add the 'uses-permission' tag to app/src/main/AndroidManifest.xml.
    static void requestStoragePermissions(Activity activity)
    {
        int permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);

        if (PackageManager.PERMISSION_GRANTED != permission)
            ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
    }

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
}
