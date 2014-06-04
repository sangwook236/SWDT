using System;
using System.Collections.Generic;
using System.Text;

namespace mathematics.math_net
{
    using MathNet.Numerics.LinearAlgebra;

    class MatrixOperation
    {
        public static void run()
        {
            runBasicOperation();
        }

        static void runBasicOperation()
        {
            Random rand = new Random();

            int row1 = 5, col1 = 5;
            double[][] data1 = new double[row1][];
            for (int i = 0; i < row1; ++i)
            {
                data1[i] = new double[col1];
                for (int j = 0; j < col1; ++j)
                {
                    data1[i][j] = rand.NextDouble() * 100.0;
                }
            }

            Matrix A1 = new Matrix(data1);
            //Matrix A1 = Matrix.Random(row1, col1);

            double[] data2 = { 0.0, 2.0, 4.0, 1.0, 3.0, 5.0 };
            int row2 = 3;
            Matrix A2 = new Matrix(data2, row2);
            Console.WriteLine("matrix = {0}", A2.ToString());
        }
   }
}
