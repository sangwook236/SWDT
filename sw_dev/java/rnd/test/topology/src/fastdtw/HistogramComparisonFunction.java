package fastdtw;

import com.util.DistanceFunction;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class HistogramComparisonFunction implements DistanceFunction {

	private static final int dims_ = 1;
	private static final int[] dim_sizes_ = { 360 };
	private static final float[][] ranges_ = { { 0, 359 } };
	private static final int uniform_ = 1;

	private static CvScalar val1_;
	private static CvScalar val2_;

	private static CvHistogram histo1_;
	private static CvHistogram histo2_;

	static {
		histo1_ = cvCreateHist(dims_, dim_sizes_, CV_HIST_ARRAY, ranges_, uniform_);
		histo2_ = cvCreateHist(dims_, dim_sizes_, CV_HIST_ARRAY, ranges_, uniform_);
		val1_ = new CvScalar();
		val2_ = new CvScalar();
	}

	public HistogramComparisonFunction()
	{
	}

	public void finalize()
	{
		// FIXME [modify] >> clean-up process is required.
        // Clean-up.
		//cvReleaseHist(histo1_);
		//cvReleaseHist(histo2_);
		//histo1_ = null;
		//histo2_ = null;
	}

	public double calcDistance(double[] vector1, double[] vector2)
	{
		if (vector1.length != vector2.length)
			throw new InternalError("ERROR: cannot calculate the distance between vectors of different sizes.");

		CvMatND mat1 = histo1_.mat();
		CvMatND mat2 = histo2_.mat();
		for (int i = 0; i < vector1.length; ++i)
		{
			val1_.setVal(0, vector1[i]);
			val2_.setVal(0, vector2[i]);
			cvSet1D(mat1, i, val1_);
			cvSet1D(mat2, i, val2_);
		}

		// TODO [check] >> zero histogram is treated as an uniform distribution.
		final double eps = 1.0e-20;
		final CvScalar sum1 = cvSum(mat1);
		if (Math.abs(sum1.val(0)) < eps)
		{
			for (int i = 0; i < vector1.length; ++i)
			{
				val1_.setVal(0, 1.0);
				cvSet1D(mat1, i, val1_);
			}
		}
		final CvScalar sum2 = cvSum(mat2);
		if (Math.abs(sum2.val(0)) < eps)
		{
			for (int i = 0; i < vector2.length; ++i)
			{
				val2_.setVal(0, 1.0);
				cvSet1D(mat2, i, val2_);
			}
		}

		cvNormalizeHist(histo1_, 1.0);
		cvNormalizeHist(histo2_, 1.0);

		//final double dist = cvCompareHist(histo1_, histo2_, CV_COMP_CHISQR);
		final double dist = cvCompareHist(histo1_, histo2_, CV_COMP_BHATTACHARYYA);

		return dist;
	}

}
