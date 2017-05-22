package fastdtw;

import com.timeseries.TimeSeries;
import com.util.DistanceFunction;
import com.util.DistanceFunctionFactory;
import com.dtw.TimeWarpInfo;

public class DTWExample {
	
	// [ref] ${FASTDTW_HOME}/src/com/DtwTest.java. 
	public static void run(String[] args)
	{
		final String filename1 = "./data/distance_measure/trace0.csv";
		final String filename2 = "./data/distance_measure/trace1.csv";

		final TimeSeries tsI = new TimeSeries(filename1, false, false, ',');
        final TimeSeries tsJ = new TimeSeries(filename2, false, false, ',');
        
        final DistanceFunction distFn = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance. 
        
        final TimeWarpInfo info = com.dtw.DTW.getWarpInfoBetween(tsI, tsJ, distFn);

        System.out.println("Warp Distance: " + info.getDistance());
        System.out.println("Warp Path:     " + info.getPath());
	}

}
