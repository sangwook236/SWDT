package javacv;

import static org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.indexer.*;

public class JavaCV_MatrixOperation {

	public static void run(String[] args)
	{
		final int rows = 10, cols = 5;
		final Mat mat = new Mat(rows, cols, CV_64FC1);

		final DoubleRawIndexer indexer = mat.createIndexer();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
            	//cvSet2D(mat, i, j, cvScalar(i * 100.0 + j, 0.0, 0.0, 0.0));
            	indexer.put(i, j, (byte)(i * 100.0 + j));
            }
        
        System.out.println(mat);
	}
	
}
