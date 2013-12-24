using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using OxyPlot;
using OxyPlot.Series;

namespace visualization.qxyplot
{
    class PieSeriesExample
    {
        public static void run(string[] args)
        {
            runWorldPopulation();
        }

        // [ref] ${OxyPlot_HOME}/NET45/Examples/ExampleBrowser/ExampleBrowser.exe.
        static void runWorldPopulation()
        {
            var plotModel = new PlotModel();
            plotModel.Title = "World population by continent";

            var pieSeries1 = new PieSeries();
            pieSeries1.InsideLabelPosition = 0.8;
            pieSeries1.StrokeThickness = 2;
            pieSeries1.Slices.Add(new PieSlice("Africa", 15.0));
            pieSeries1.Slices.Add(new PieSlice("Americas", 13.0));
            pieSeries1.Slices.Add(new PieSlice("Asia", 60.0));
            pieSeries1.Slices.Add(new PieSlice("Europe", 11.0));
            pieSeries1.Slices.Add(new PieSlice("Oceania", 1.0));
            plotModel.Series.Add(pieSeries1);

            //
            String output_filename = "../data/visualization/qxyplot/pieseries_world_population.pdf";
            using (var stream = File.Create(output_filename))
            {
                PdfExporter.Export(plotModel, stream, 600, 400);
            }
        }
    }
}
