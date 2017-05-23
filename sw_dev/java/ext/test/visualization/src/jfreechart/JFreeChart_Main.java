package jfreechart;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.title.TextTitle;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.data.time.Month;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.ui.HorizontalAlignment;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.RefineryUtilities;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.Point;
import java.awt.RadialGradientPaint;
import java.awt.geom.Point2D;
import javax.swing.JPanel;
import java.text.SimpleDateFormat;

public class JFreeChart_Main {

	public static void run(String[] args)
	{
		//SimplePieChartExample();

		BarChartDemo1();
		//PieChartDemo1();
		//TimeSeriesChartDemo1();
	}

	// REF [doc] >> JFreeChart Install Manual.
	private static void SimplePieChartExample()
	{
		// Create a dataset.
		DefaultPieDataset data = new DefaultPieDataset();
		data.setValue("Category 1", 43.2);
		data.setValue("Category 2", 27.9);
		data.setValue("Category 3", 79.5);

		// Create a chart.
		JFreeChart chart = ChartFactory.createPieChart(
			"Sample Pie Chart",
			data,
			true,  // legend?
			true,  // tooltips?
			false  // URLs?
		);

		// Create and display a frame.
		ChartFrame frame = new ChartFrame("Simple Pie Chart Example", chart);
		frame.pack();
		frame.setVisible(true);
	}

	// REF [file] >> ${JFREECHART_HOME}/source/org/jfree/chart/demo/BarChartDemo1.java.
	private static void BarChartDemo1()
	{
		// Crerate a sample dataset.
		// Row keys.
		String series1 = "First";
		String series2 = "Second";
		String series3 = "Third";

		// Column keys.
		String category1 = "Category 1";
		String category2 = "Category 2";
		String category3 = "Category 3";
		String category4 = "Category 4";
		String category5 = "Category 5";

		// Create the dataset.
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();

		dataset.addValue(1.0, series1, category1);
		dataset.addValue(4.0, series1, category2);
		dataset.addValue(3.0, series1, category3);
		dataset.addValue(5.0, series1, category4);
		dataset.addValue(5.0, series1, category5);

		dataset.addValue(5.0, series2, category1);
		dataset.addValue(7.0, series2, category2);
		dataset.addValue(6.0, series2, category3);
		dataset.addValue(8.0, series2, category4);
		dataset.addValue(4.0, series2, category5);

		dataset.addValue(4.0, series3, category1);
		dataset.addValue(3.0, series3, category2);
		dataset.addValue(2.0, series3, category3);
		dataset.addValue(3.0, series3, category4);
		dataset.addValue(6.0, series3, category5);

		//
		// Create the chart...
		JFreeChart chart = ChartFactory.createBarChart(
		    "Bar Chart Demo 1",  // Chart title.
		    "Category",  // Domain axis label.
		    "Value",  // Range axis label.
		    dataset,  // Data.
		    PlotOrientation.VERTICAL,  // Orientation.
		    true,  // Legend?
		    true,  // Tooltips?
		    false  // URLs?
		);

		// NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART.

		// Set the background color for the chart.
		chart.setBackgroundPaint(Color.white);

		// Get a reference to the plot for further customization.
		CategoryPlot plot = (CategoryPlot)chart.getPlot();

		// Set the range axis to display integers only.
		NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

		// Disable bar outlines.
		BarRenderer renderer = (BarRenderer) plot.getRenderer();
		renderer.setDrawBarOutline(false);

		// Set up gradient paints for series.
		GradientPaint gp0 = new GradientPaint(0.0f, 0.0f, Color.blue, 0.0f, 0.0f, new Color(0, 0, 64));
		GradientPaint gp1 = new GradientPaint(0.0f, 0.0f, Color.green, 0.0f, 0.0f, new Color(0, 64, 0));
		GradientPaint gp2 = new GradientPaint(0.0f, 0.0f, Color.red, 0.0f, 0.0f, new Color(64, 0, 0));
		renderer.setSeriesPaint(0, gp0);
		renderer.setSeriesPaint(1, gp1);
		renderer.setSeriesPaint(2, gp2);

		CategoryAxis domainAxis = plot.getDomainAxis();
		domainAxis.setCategoryLabelPositions(
		        CategoryLabelPositions.createUpRotationLabelPositions(Math.PI / 6.0)
		);

		// Create a panel.
	    chart.setPadding(new RectangleInsets(4, 8, 2, 2));
	    ChartPanel panel = new ChartPanel(chart);
	    panel.setFillZoomRectangle(true);
	    panel.setMouseWheelEnabled(true);
	    panel.setPreferredSize(new Dimension(600, 300));

		// Create and display a frame.
		ChartFrame frame = new ChartFrame("Bar Chart Demo 1", chart);
		frame.setContentPane(panel);
		frame.pack();
		RefineryUtilities.centerFrameOnScreen(frame);
		frame.setVisible(true);
	}

	// REF [file] >> ${JFREECHART_HOME}/source/org/jfree/chart/demo/PieChartDemo1.java.
	private static void PieChartDemo1()
	{
		// Set a theme using the new shadow generator feature available in 1.0.14
		// - For backwards compatibility it is not enabled by default.
		ChartFactory.setChartTheme(new StandardChartTheme("JFree/Shadow", true));

		// Create a sample dataset.
	    DefaultPieDataset dataset = new DefaultPieDataset();
	    dataset.setValue("Samsung", new Double(27.8));
	    dataset.setValue("Others", new Double(55.3));
	    dataset.setValue("Nokia", new Double(16.8));
	    dataset.setValue("Apple", new Double(17.1));

	    JFreeChart chart = ChartFactory.createPieChart(
	    	"Smart Phones Manufactured / Q3 2011",  // Chart title.
	    	dataset,  // Data.
	    	false,  // Legend?
	    	true,  // Tooltips?
	    	false  // URL generation?
	    );

		// Set a custom background for the chart.
		chart.setBackgroundPaint(
			new GradientPaint(new Point(0, 0), new Color(20, 20, 20), new Point(400, 200), Color.DARK_GRAY)
		);

		// Customize the title position and font.
		TextTitle t = chart.getTitle();
		t.setHorizontalAlignment(HorizontalAlignment.LEFT);
		t.setPaint(new Color(240, 240, 240));
		t.setFont(new Font("Arial", Font.BOLD, 26));
		  
		PiePlot plot = (PiePlot)chart.getPlot();
		plot.setBackgroundPaint(null);
		plot.setInteriorGap(0.04);
		plot.setOutlineVisible(false);

		// Use gradients and white borders for the section colors.
		plot.setSectionPaint("Others", createGradientPaint(new Color(200, 200, 255), Color.BLUE));
		plot.setSectionPaint("Samsung", createGradientPaint(new Color(255, 200, 200), Color.RED));
		plot.setSectionPaint("Apple", createGradientPaint(new Color(200, 255, 200), Color.GREEN));
		plot.setSectionPaint("Nokia", createGradientPaint(new Color(200, 255, 200), Color.YELLOW));
		plot.setBaseSectionOutlinePaint(Color.WHITE);
		plot.setSectionOutlinesVisible(true);
		plot.setBaseSectionOutlineStroke(new BasicStroke(2.0f));

		// Customize the section label appearance.
		plot.setLabelFont(new Font("Courier New", Font.BOLD, 20));
		plot.setLabelLinkPaint(Color.WHITE);
		plot.setLabelLinkStroke(new BasicStroke(2.0f));
		plot.setLabelOutlineStroke(null);
		plot.setLabelPaint(Color.WHITE);
		plot.setLabelBackgroundPaint(null);

		// Add a subtitle giving the data source.
		TextTitle source = new TextTitle(
			"Source: http://www.bbc.co.uk/news/business-15489523", 
			new Font("Courier New", Font.PLAIN, 12)
		);
		source.setPaint(Color.WHITE);
		source.setPosition(RectangleEdge.BOTTOM);
		source.setHorizontalAlignment(HorizontalAlignment.RIGHT);
		chart.addSubtitle(source);
 
		// Create a panel.
	    chart.setPadding(new RectangleInsets(4, 8, 2, 2));
	    ChartPanel panel = new ChartPanel(chart);
	    panel.setMouseWheelEnabled(true);
	    panel.setPreferredSize(new Dimension(600, 300));

		// Create and display a frame.
		ChartFrame frame = new ChartFrame("Pie Chart Demo 1", chart);
		frame.setContentPane(panel);
		frame.pack();
		RefineryUtilities.centerFrameOnScreen(frame);
		frame.setVisible(true);
	}

	private static RadialGradientPaint createGradientPaint(Color c1, Color c2)
	{
		Point2D center = new Point2D.Float(0, 0);
		float radius = 200;
		float[] dist = { 0.0f, 1.0f };
		return new RadialGradientPaint(center, radius, dist, new Color[] { c1, c2 });
	}

	// REF [file] >> ${JFREECHART_HOME}/source/org/jfree/chart/demo/TimeSeriesChartDemo1.java.
	private static void TimeSeriesChartDemo1()
	{
		// Create a dataset, consisting of two series of monthly data.
		TimeSeries s1 = new TimeSeries("L&G European Index Trust");
		s1.add(new Month(2, 2001), 181.8);
		s1.add(new Month(3, 2001), 167.3);
		s1.add(new Month(4, 2001), 153.8);
		s1.add(new Month(5, 2001), 167.6);
		s1.add(new Month(6, 2001), 158.8);
		s1.add(new Month(7, 2001), 148.3);
		s1.add(new Month(8, 2001), 153.9);
		s1.add(new Month(9, 2001), 142.7);
		s1.add(new Month(10, 2001), 123.2);
		s1.add(new Month(11, 2001), 131.8);
		s1.add(new Month(12, 2001), 139.6);
		s1.add(new Month(1, 2002), 142.9);
		s1.add(new Month(2, 2002), 138.7);
		s1.add(new Month(3, 2002), 137.3);
		s1.add(new Month(4, 2002), 143.9);
		s1.add(new Month(5, 2002), 139.8);
		s1.add(new Month(6, 2002), 137.0);
		s1.add(new Month(7, 2002), 132.8);

		TimeSeries s2 = new TimeSeries("L&G UK Index Trust");
		s2.add(new Month(2, 2001), 129.6);
		s2.add(new Month(3, 2001), 123.2);
		s2.add(new Month(4, 2001), 117.2);
		s2.add(new Month(5, 2001), 124.1);
		s2.add(new Month(6, 2001), 122.6);
		s2.add(new Month(7, 2001), 119.2);
		s2.add(new Month(8, 2001), 116.5);
		s2.add(new Month(9, 2001), 112.7);
		s2.add(new Month(10, 2001), 101.5);
		s2.add(new Month(11, 2001), 106.1);
		s2.add(new Month(12, 2001), 110.3);
		s2.add(new Month(1, 2002), 111.7);
		s2.add(new Month(2, 2002), 111.0);
		s2.add(new Month(3, 2002), 109.6);
		s2.add(new Month(4, 2002), 113.2);
		s2.add(new Month(5, 2002), 111.6);
		s2.add(new Month(6, 2002), 108.8);
		s2.add(new Month(7, 2002), 101.6);

		TimeSeriesCollection dataset = new TimeSeriesCollection();
		dataset.addSeries(s1);
		dataset.addSeries(s2);

		// Create a chart.
		JFreeChart chart = ChartFactory.createTimeSeriesChart(
		    "Legal & General Unit Trust Prices",  // Chart title.
		    "Date",  // x-axis label.
		    "Price Per Unit",  // y-axis label.
		    dataset,  // Data.
		    true,  // Legend?
		    true,  // Tooltips?
		    false  // URLs?
		);

		chart.setBackgroundPaint(Color.white);

		XYPlot plot = (XYPlot)chart.getPlot();
		plot.setBackgroundPaint(Color.lightGray);
		plot.setDomainGridlinePaint(Color.white);
		plot.setRangeGridlinePaint(Color.white);
		plot.setAxisOffset(new RectangleInsets(5.0, 5.0, 5.0, 5.0));
		plot.setDomainCrosshairVisible(true);
		plot.setRangeCrosshairVisible(true);

		XYItemRenderer r = plot.getRenderer();
		if (r instanceof XYLineAndShapeRenderer)
		{
		    XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer)r;
		    renderer.setBaseShapesVisible(true);
		    renderer.setBaseShapesFilled(true);
		    renderer.setDrawSeriesLineAsPath(true);
		}

		DateAxis axis = (DateAxis)plot.getDomainAxis();
		axis.setDateFormatOverride(new SimpleDateFormat("MMM-yyyy"));

		// Create a panel.
	    ChartPanel panel = new ChartPanel(chart);
	    panel.setFillZoomRectangle(true);
	    panel.setMouseWheelEnabled(true);

		// Create and display a frame.
		ChartFrame frame = new ChartFrame("Time Series Chart Demo 1", chart);
		frame.setContentPane(panel);
		frame.pack();
		RefineryUtilities.centerFrameOnScreen(frame);
		frame.setVisible(true);
	}

}
