package javacv;

import static com.googlecode.javacv.cpp.opencv_core.*;

public class JavaCV_MatrixOperation {

	public static void run(String[] args)
	{
		final int rows = 10, cols = 5;
        final CvMat mat = cvCreateMat(rows, cols, CV_64FC1);
        
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
            {
            	//cvSet2D(mat, i, j, cvScalar(i * 100.0 + j, 0.0, 0.0, 0.0));
        		cvSetReal2D(mat, i, j, i * 100.0 + j);
            }
        
        System.out.println(mat);
        
        // Clean-up.
        cvReleaseMat(mat);
	}
	
}
