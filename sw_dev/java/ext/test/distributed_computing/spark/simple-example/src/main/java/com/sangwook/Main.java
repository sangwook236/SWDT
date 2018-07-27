package com.sangwook;

import com.sangwook.*;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		try
		{
			System.out.println("Spark ---------------------------------------------------------------");
			com.sangwook.spark.Spark_Main.run(args);
		}
		catch (Exception ex)
		{
			System.err.println("Exception occurred: " + ex.toString());
		}
	}

}
