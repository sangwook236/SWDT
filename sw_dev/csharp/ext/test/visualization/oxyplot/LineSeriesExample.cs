using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace visualization.oxyplot
{
    class LineSeriesExample
    {
        public static void run(string[] args)
        {
            runMarketTypes();
        }

        // [ref] ${OxyPlot_HOME}/NET45/Examples/ExampleBrowser/ExampleBrowser.exe.
        static void runMarketTypes()
        {
            var plotModel = new PlotModel();
            plotModel.Title = "Marker types";

            var linearAxis1 = new LinearAxis();
            linearAxis1.Position = AxisPosition.Bottom;
            plotModel.Axes.Add(linearAxis1);

            var linearAxis2 = new LinearAxis();
            plotModel.Axes.Add(linearAxis2);

            var lineSeries1 = new LineSeries();
            lineSeries1.MarkerStroke = OxyColors.Black;
            lineSeries1.MarkerType = MarkerType.Circle;
            lineSeries1.Title = "Circle";
            lineSeries1.Points.Add(new DataPoint(11.8621869598805, 1.4144100893356));
            lineSeries1.Points.Add(new DataPoint(18.1636718735861, 3.34725344383496));
            lineSeries1.Points.Add(new DataPoint(26.1161091961507, 5.10126935043431));
            lineSeries1.Points.Add(new DataPoint(29.99451482482, 6.99704349646207));
            lineSeries1.Points.Add(new DataPoint(41.9055463103138, 8.21414295500803));
            lineSeries1.Points.Add(new DataPoint(48.6674896314123, 9.55912132400978));
            lineSeries1.Points.Add(new DataPoint(56.0143694207139, 10.8773872386093));
            lineSeries1.Points.Add(new DataPoint(67.4011132397694, 12.2735980540857));
            lineSeries1.Points.Add(new DataPoint(74.2207925069243, 13.5990719658318));
            lineSeries1.Points.Add(new DataPoint(78.4307891402537, 14.9570749639334));
            plotModel.Series.Add(lineSeries1);

            var lineSeries2 = new LineSeries();
            lineSeries2.MarkerStroke = OxyColors.Black;
            lineSeries2.MarkerType = MarkerType.Cross;
            lineSeries2.Title = "Cross";
            lineSeries2.Points.Add(new DataPoint(10.2551460425626, 1.12034707615168));
            lineSeries2.Points.Add(new DataPoint(17.2749783542356, 2.66979351810636));
            lineSeries2.Points.Add(new DataPoint(21.436097549012, 4.12809186853845));
            lineSeries2.Points.Add(new DataPoint(30.78668166268, 5.20108992848596));
            lineSeries2.Points.Add(new DataPoint(33.2050538869598, 6.637421580794));
            lineSeries2.Points.Add(new DataPoint(45.0304378052384, 7.66540479830718));
            lineSeries2.Points.Add(new DataPoint(56.7345610422709, 9.54063240976102));
            lineSeries2.Points.Add(new DataPoint(66.1928816270981, 10.7854569441571));
            lineSeries2.Points.Add(new DataPoint(74.2900140398601, 12.665478166037));
            lineSeries2.Points.Add(new DataPoint(79.0780252865879, 13.7934278663217));
            plotModel.Series.Add(lineSeries2);

            var lineSeries3 = new LineSeries();
            lineSeries3.MarkerStroke = OxyColors.Black;
            lineSeries3.MarkerType = MarkerType.Diamond;
            lineSeries3.Title = "Diamond";
            lineSeries3.Points.Add(new DataPoint(8.23705901030314, 1.04818704773122));
            lineSeries3.Points.Add(new DataPoint(15.3159614062477, 2.61679908755086));
            lineSeries3.Points.Add(new DataPoint(26.0893045543178, 3.88729132986036));
            lineSeries3.Points.Add(new DataPoint(30.3871664714008, 5.60047892089955));
            lineSeries3.Points.Add(new DataPoint(42.2880761336014, 6.83121995620021));
            lineSeries3.Points.Add(new DataPoint(45.5782417531955, 8.70534298555243));
            lineSeries3.Points.Add(new DataPoint(53.6039859063942, 9.76094881480604));
            lineSeries3.Points.Add(new DataPoint(63.170427156226, 11.6268204346424));
            lineSeries3.Points.Add(new DataPoint(74.2946957379089, 13.5075224840583));
            lineSeries3.Points.Add(new DataPoint(83.4620100881262, 14.6444678658827));
            plotModel.Series.Add(lineSeries3);

            var lineSeries4 = new LineSeries();
            lineSeries4.MarkerStroke = OxyColors.Black;
            lineSeries4.MarkerType = MarkerType.Plus;
            lineSeries4.Title = "Plus";
            lineSeries4.Points.Add(new DataPoint(2.18957495232559, 1.11484250291942));
            lineSeries4.Points.Add(new DataPoint(12.7289441883233, 2.66961399916076));
            lineSeries4.Points.Add(new DataPoint(17.0745548555043, 4.11782511375743));
            lineSeries4.Points.Add(new DataPoint(22.8202153084894, 5.77728548123375));
            lineSeries4.Points.Add(new DataPoint(30.5445538696575, 7.4711353287432));
            lineSeries4.Points.Add(new DataPoint(40.409529611659, 9.07308203357881));
            lineSeries4.Points.Add(new DataPoint(42.8762574823928, 10.166131996627));
            lineSeries4.Points.Add(new DataPoint(52.219228453105, 11.9434595531521));
            lineSeries4.Points.Add(new DataPoint(60.3437968186772, 13.8768307444997));
            lineSeries4.Points.Add(new DataPoint(69.0649180901539, 15.8224504216679));
            plotModel.Series.Add(lineSeries4);

            var lineSeries5 = new LineSeries();
            lineSeries5.MarkerStroke = OxyColors.Black;
            lineSeries5.MarkerType = MarkerType.Square;
            lineSeries5.Title = "Square";
            lineSeries5.Points.Add(new DataPoint(4.28512158723787, 1.0218708203276));
            lineSeries5.Points.Add(new DataPoint(7.11419252451239, 2.83296700745493));
            lineSeries5.Points.Add(new DataPoint(12.1873434279986, 3.94138236993057));
            lineSeries5.Points.Add(new DataPoint(18.4414314499318, 5.85618886438021));
            lineSeries5.Points.Add(new DataPoint(21.6272663146384, 7.73614930302657));
            lineSeries5.Points.Add(new DataPoint(26.9512430769164, 9.46516049488688));
            lineSeries5.Points.Add(new DataPoint(30.584140945498, 10.6070162377353));
            lineSeries5.Points.Add(new DataPoint(33.6740629960196, 12.1158796358462));
            lineSeries5.Points.Add(new DataPoint(37.6165642373341, 14.0689983791993));
            lineSeries5.Points.Add(new DataPoint(42.9570739683495, 15.4981215794096));
            plotModel.Series.Add(lineSeries5);
            
            var lineSeries6 = new LineSeries();
            lineSeries6.MarkerStroke = OxyColors.Black;
            lineSeries6.MarkerType = MarkerType.Star;
            lineSeries6.Title = "Star";
            lineSeries6.Points.Add(new DataPoint(10.9592619701099, 1.57032582423199));
            lineSeries6.Points.Add(new DataPoint(14.3667993193338, 2.86656694294259));
            lineSeries6.Points.Add(new DataPoint(20.8092339303387, 4.3936368000664));
            lineSeries6.Points.Add(new DataPoint(31.0837363531272, 5.90316125233805));
            lineSeries6.Points.Add(new DataPoint(36.2968236749511, 7.88247782079618));
            lineSeries6.Points.Add(new DataPoint(40.8309715077425, 9.86153348761682));
            lineSeries6.Points.Add(new DataPoint(44.9168707304247, 11.6326452454704));
            lineSeries6.Points.Add(new DataPoint(56.0012029614305, 13.6297319203754));
            lineSeries6.Points.Add(new DataPoint(58.3205570533502, 14.6725726568478));
            lineSeries6.Points.Add(new DataPoint(62.7951211122773, 15.7987183610903));
            plotModel.Series.Add(lineSeries6);
            
            var lineSeries7 = new LineSeries();
            lineSeries7.MarkerStroke = OxyColors.Black;
            lineSeries7.MarkerType = MarkerType.Triangle;
            lineSeries7.Title = "Triangle";
            lineSeries7.Points.Add(new DataPoint(2.2280231240336, 1.45975955084886));
            lineSeries7.Points.Add(new DataPoint(9.6367919340901, 3.15223296831932));
            lineSeries7.Points.Add(new DataPoint(15.2513136469067, 4.20971935065916));
            lineSeries7.Points.Add(new DataPoint(21.6378828266812, 6.11453639488413));
            lineSeries7.Points.Add(new DataPoint(33.4784604066417, 7.33997009384445));
            lineSeries7.Points.Add(new DataPoint(41.3092347305777, 8.99930818704856));
            lineSeries7.Points.Add(new DataPoint(49.3024369130388, 10.2422971139859));
            lineSeries7.Points.Add(new DataPoint(51.7993717146103, 11.9004834484777));
            lineSeries7.Points.Add(new DataPoint(62.6105425686625, 13.6871494234945));
            lineSeries7.Points.Add(new DataPoint(68.2790698289308, 15.4673946567194));
            plotModel.Series.Add(lineSeries7);

            //
#if false
            String output_filename = "../data/visualization/oxyplot/lineseries_marker_type.png";
            using (var stream = File.Create(output_filename))
            {
                System.Windows.Threading.Dispatcher.CurrentDispatcher.Invoke(
                    System.Windows.Threading.DispatcherPriority.Normal,
                    new Action(
                        delegate
                        {
                            OxyPlot.Wpf.PngExporter.Export(plotModel, stream, 600, 400, OxyColors.White);  // run-time error.
                        }
                    )
                );
            }
#elif true
            String output_filename = "../data/visualization/oxyplot/lineseries_marker_type.pdf";
            using (var stream = File.Create(output_filename))
            {
                PdfExporter.Export(plotModel, stream, 600, 400);
            }
#elif true
            // Copy to clipboard.
            using (var stream = new MemoryStream())
            {
                OxyPlot.Wpf.PngExporter.Export(plotModel, stream, 600, 400, OxyColors.White);  // run-time error.
            }
#endif
        }
    }
}
