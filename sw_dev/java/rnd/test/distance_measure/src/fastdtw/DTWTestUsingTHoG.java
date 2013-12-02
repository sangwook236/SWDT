package fastdtw;

import com.dtw.TimeWarpInfo;
import com.timeseries.TimeSeries;
import com.timeseries.TimeSeriesPoint;
import com.util.DistanceFunction;
import com.util.DistanceFunctionFactory;

public class DTWTestUsingTHoG {
	
	public static void run(String[] args)
	{
		//compareFullTHoGs();
		//comparePartialTHoGs();
		//compareTHoGsUsingFullReferenceTHoG();
		compareTHoGsUsingPartialReferenceTHoG();
	}

	private static void compareFullTHoGs()
	{
		final String[] filenameList = {
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_1.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_2.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_4.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_7.HoG",
		};

		final TimeSeries ts1 = new TimeSeries(filenameList[0], false, false, ' ');
		final TimeSeries ts2 = new TimeSeries(filenameList[1], false, false, ' ');
		
        //final DistanceFunction distFunc = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        final DistanceFunction distFunc = new HistogramComparisonFunction();
        final int radius = 10;
    	final double startTime = (double)System.nanoTime() * 1.0e-6;
        //final TimeWarpInfo info = com.dtw.DTW.getWarpInfoBetween(ts1, ts2, distFunc);
        final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(ts1, ts2, radius, distFunc);
    	final double endTime = (double)System.nanoTime() * 1.0e-6;
		
    	System.out.println("Elapsed time:  " + (endTime - startTime));
        System.out.println("Warp distance: " + info.getDistance());
        System.out.println("Warp path:     " + info.getPath());
	}

	private static void comparePartialTHoGs()
	{
		final String[] filenameList = {
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_1.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_2.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_4.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_7.HoG",
		};

		final TimeSeries ts1 = new TimeSeries(filenameList[0], false, false, ' ');
		final TimeSeries ts2 = new TimeSeries(filenameList[0], false, false, ' ');
		
    	final int numFeatures1 = ts1.numOfDimensions();
    	final int numFrames1 = ts1.numOfPts();
    	final int numFeatures2 = ts2.numOfDimensions();
    	final int numFrames2 = ts2.numOfPts();

    	final int frameStart1 = 0, frameEnd1 = 20;
    	final int frameStart2 = 0, frameEnd2 = 30;

    	final TimeSeries ts1_partial = new TimeSeries(numFeatures1);
    	ts1_partial.setLabels(ts1.getLabels());
    	for (int i = frameStart1; i <= frameEnd1 && i < numFrames1; ++i)
    		ts1_partial.addLast((double)(i - frameStart1), new TimeSeriesPoint(ts1.getMeasurementVector(i)));
    	final TimeSeries ts2_partial = new TimeSeries(numFeatures2);
    	ts2_partial.setLabels(ts2.getLabels());
    	for (int i = frameStart2; i <= frameEnd2 && i < numFrames2; ++i)
    		ts2_partial.addLast((double)(i - frameStart2), new TimeSeriesPoint(ts2.getMeasurementVector(i)));

    	//final DistanceFunction distFunc = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        final DistanceFunction distFunc = new HistogramComparisonFunction();
        final int radius = 10;
    	final double startTime = (double)System.nanoTime() * 1.0e-6;
        //final TimeWarpInfo info = com.dtw.DTW.getWarpInfoBetween(ts1_partial, ts2_partial, distFunc);
        final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(ts1_partial, ts2_partial, radius, distFunc);
    	final double endTime = (double)System.nanoTime() * 1.0e-6;
		
    	System.out.println("Elapsed time:  " + (endTime - startTime));
        System.out.println("Warp distance: " + info.getDistance());
        System.out.println("Warp path:     " + info.getPath());
	}
	
	private static void compareTHoGsUsingFullReferenceTHoG()
	{
		final String[] filenameList = {
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_1.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_2.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_4.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_7.HoG",
		};

		final TimeSeries[] timeSeriesList = new TimeSeries [filenameList.length];
		for (int i = 0; i < filenameList.length; ++i)
			timeSeriesList[i] = new TimeSeries(filenameList[i], false, false, ' ');

		//
		double[][][] result = new double [filenameList.length][filenameList.length][];
		
        //final DistanceFunction distFunc = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        final DistanceFunction distFunc = new HistogramComparisonFunction();
        final int radius = 10;
    	final int frameWinSize = 15;
        for (int tt = 0; tt < timeSeriesList.length; ++tt)
		{
        	final TimeSeries ts1 = timeSeriesList[tt];
        	//final int numFeatures1 = ts1.numOfDimensions();
        	//final int numFrames1 = ts1.numOfPts();

	        for (int uu = 0; uu < timeSeriesList.length; ++uu)
			{
		        System.out.println("(" + tt + ", " + uu + ") is processing ...");

		        final TimeSeries ts2 = timeSeriesList[uu];
	        	final int numFeatures2 = ts2.numOfDimensions();
	        	final int numFrames2 = ts2.numOfPts();
	        
	        	result[tt][uu] = new double [numFrames2 - frameWinSize + 1];
		        for (int ff = 0; ff <= numFrames2 - frameWinSize; ++ff)
				{
		        	final int frameStart = ff, frameEnd = ff + frameWinSize - 1;
		        	
		        	final TimeSeries ts2_partial = new TimeSeries(numFeatures2);
		        	ts2_partial.setLabels(ts2.getLabels());
		        	for (int i = frameStart; i <= frameEnd && i < numFrames2; ++i)
		        		ts2_partial.addLast((double)(i - frameStart), new TimeSeriesPoint(ts2.getMeasurementVector(i)));

		        	//
		        	final double startTime = (double)System.nanoTime() * 1.0e-6;
			        //final TimeWarpInfo info = com.dtw.DTW.getWarpInfoBetween(ts1, ts2_partial, distFunc);
			        final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(ts1, ts2_partial, radius, distFunc);
		        	final double endTime = (double)System.nanoTime() * 1.0e-6;

		        	//
		        	System.out.println("\tElapsed time:  " + (endTime - startTime));
			        System.out.println("\tWarp distance: " + info.getDistance());
			        System.out.println("\tWarp path:     " + info.getPath());
			        
			        result[tt][uu][ff] = info.getDistance();
				}
			}
		}
        
        // Display result.
        for (int i = 0; i < result.length; ++i)
        {
            for (int j = 0; j < result[i].length; ++j)
            {
                for (int k = 0; k < result[i][j].length; ++k)
                	System.out.print(result[i][j][k] + ", ");
                System.out.println();
            }
        }
	}
	
	private static void compareTHoGsUsingPartialReferenceTHoG()
	{
		final String[] filenameList = {
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_1.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_2.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_4.HoG",
			"E:/dataset/motion/ChaLearn_Gesture_Challenge_dataset/quasi_lossless_format/train_data/devel16_thog3/M_7.HoG",
		};

		final TimeSeries[] timeSeriesList = new TimeSeries [filenameList.length];
		for (int i = 0; i < filenameList.length; ++i)
			timeSeriesList[i] = new TimeSeries(filenameList[i], false, false, ' ');

		//
		double[][][] result = new double [filenameList.length][filenameList.length][];
		
        //final DistanceFunction distFunc = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        final DistanceFunction distFunc = new HistogramComparisonFunction();
        final int radius = 10;
    	final int frameWinSize1 = 15;
    	final int frameWinSize2 = 15;
    	final TimeSeries ts1_partial = new TimeSeries(0);
    	final TimeSeries ts2_partial = new TimeSeries(0);
        for (int tt = 0; tt < timeSeriesList.length; ++tt)
		{
        	final TimeSeries ts1 = timeSeriesList[tt];
        	//final int numFeatures1 = ts1.numOfDimensions();
        	final int numFrames1 = ts1.numOfPts();

	        for (int uu = 0; uu < timeSeriesList.length; ++uu)
			{
		        System.out.println("(" + tt + ", " + uu + ") is processing ...");

		        final TimeSeries ts2 = timeSeriesList[uu];
	        	//final int numFeatures2 = ts2.numOfDimensions();
	        	final int numFrames2 = ts2.numOfPts();
	        
	        	result[tt][uu] = new double [numFrames2 - frameWinSize2 + 1];
		        for (int ff = 0; ff <= numFrames2 - frameWinSize2; ++ff)
				{
		        	final int frameStart2 = ff, frameEnd2 = ff + frameWinSize2 - 1;

		        	ts2_partial.clear();
		        	ts2_partial.setLabels(ts2.getLabels());
		        	for (int i = frameStart2; i <= frameEnd2 && i < numFrames2; ++i)
		        		ts2_partial.addLast((double)(i - frameStart2), new TimeSeriesPoint(ts2.getMeasurementVector(i)));

		        	//
		        	TimeWarpInfo bestWarpInfo = null;
		        	final double startTime = (double)System.nanoTime() * 1.0e-6;  // [ms].
			        for (int gg = 0; gg <= numFrames1 - frameWinSize1; ++gg)
					{
			        	final int frameStart1 = gg, frameEnd1 = gg + frameWinSize1 - 1;
			        	
			        	ts1_partial.clear();
			        	ts1_partial.setLabels(ts1.getLabels());
			        	for (int i = frameStart1; i <= frameEnd1 && i < numFrames1; ++i)
			        		ts1_partial.addLast((double)(i - frameStart1), new TimeSeriesPoint(ts1.getMeasurementVector(i)));

			        	//final TimeWarpInfo info = com.dtw.DTW.getWarpInfoBetween(ts1_partial, ts2_partial, distFunc);
			        	final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(ts1_partial, ts2_partial, radius, distFunc);
				        if (null == bestWarpInfo || bestWarpInfo.getDistance() > info.getDistance())
				        	bestWarpInfo = info;
					}
		        	final double endTime = (double)System.nanoTime() * 1.0e-6;  // [ms].

		        	//
		        	System.out.println("\tElapsed time:  " + (endTime - startTime));
			        System.out.println("\tWarp distance: " + bestWarpInfo.getDistance());
			        System.out.println("\tWarp path:     " + bestWarpInfo.getPath());
			        
			        result[tt][uu][ff] = bestWarpInfo.getDistance();
			        
			        //
			        System.gc();
			        Thread.yield();
				}
			}
		}
        
        // Display result.
        for (int i = 0; i < result.length; ++i)
        {
            for (int j = 0; j < result[i].length; ++j)
            {
                for (int k = 0; k < result[i][j].length; ++k)
                	System.out.print(result[i][j][k] + ", ");
                System.out.println();
            }
        }
	}

}
