package fastdtw;

import com.dtw.TimeWarpInfo;
import com.timeseries.TimeSeries;
import com.util.DistanceFunction;
import com.util.DistanceFunctionFactory;

public class DTWTestUsingTHoG {
	
	// [ref] ${FASTDTW_HOME}/src/com/FastDtwTest.java. 
	public static void run(String[] args)
	{
		final String[] filename_list = {
				"./data/distance_measure/M_1.HoG",
				"./data/distance_measure/M_2.HoG",
				"./data/distance_measure/M_4.HoG",
				"./data/distance_measure/M_7.HoG",
		};

		final TimeSeries tsI = new TimeSeries(filename_list[1], false, false, ' ');
        final TimeSeries tsJ = new TimeSeries(filename_list[1], false, false, ' ');
        
        //final DistanceFunction distFn = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        final DistanceFunction distFn = new HistogramComparisonFunction();
        
        final int radius = 10;
        final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(tsI, tsJ, radius, distFn);

        System.out.println("Warp Distance: " + info.getDistance());
        System.out.println("Warp Path:     " + info.getPath());
	}

}
