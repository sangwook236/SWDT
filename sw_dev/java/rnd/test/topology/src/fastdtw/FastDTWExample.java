package fastdtw;

import com.timeseries.TimeSeries;
import com.util.DistanceFunction;
import com.util.DistanceFunctionFactory;
import com.dtw.TimeWarpInfo;

public class FastDTWExample {
	
	// REF [file] >> ${FASTDTW_HOME}/src/com/FastDtwTest.java. 
	public static void run(String[] args)
	{
		final String filename1 = "./data/topology/trace0.csv";
		final String filename2 = "./data/topology/trace1.csv";

		final TimeSeries tsI = new TimeSeries(filename1, false, false, ',');
        final TimeSeries tsJ = new TimeSeries(filename2, false, false, ',');
        
        final DistanceFunction distFn = DistanceFunctionFactory.getDistFnByName("EuclideanDistance");  // EuclideanDistance, ManhattanDistance, BinaryDistance.
        
        final int radius = 10;
        final TimeWarpInfo info = com.dtw.FastDTW.getWarpInfoBetween(tsI, tsJ, radius, distFn);

        System.out.println("Warp Distance: " + info.getDistance());
        System.out.println("Warp Path:     " + info.getPath());
	}

}
