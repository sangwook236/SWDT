package fastdtw;

import com.util.DistanceFunction;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;

public class HistogramComparisonFunction implements DistanceFunction {

	public HistogramComparisonFunction()
	{
	}
   
	public double calcDistance(double[] vector1, double[] vector2)
	{
		if (vector1.length != vector2.length)
			throw new InternalError("ERROR: cannot calculate the distance between vectors of different sizes.");

		final int dims = 1;
		final int[] dim_sizes = { 360 };
		final float[][] ranges = { { 0, 359 } };
		final int uniform = 1;

		CvHistogram histo1 = cvCreateHist(dims, dim_sizes, CV_HIST_ARRAY, ranges, uniform);
		CvHistogram histo2 = cvCreateHist(dims, dim_sizes, CV_HIST_ARRAY, ranges, uniform);
		
		CvMatND mat1 = histo1.mat();
		CvMatND mat2 = histo2.mat();
		for (int i = 0; i < vector1.length; ++i)
		{
			cvSet1D(mat1, i, cvScalar(vector1[i], 0.0, 0.0, 0.0));
			cvSet1D(mat2, i, cvScalar(vector2[i], 0.0, 0.0, 0.0));
		}
		
		cvNormalizeHist(histo1, 1.0);
		cvNormalizeHist(histo2, 1.0);

		final double dist = cvCompareHist(histo1, histo2, CV_COMP_CHISQR);
		//final double dist = cvCompareHist(histo1, histo2, CV_COMP_BHATTACHARYYA);

		return dist;
	}
	
}
