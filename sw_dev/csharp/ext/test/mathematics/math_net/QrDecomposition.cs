using System;
using System.Collections.Generic;
using System.Text;

namespace mathematics.math_net
{
    using MathNet.Numerics.LinearAlgebra;

    class QrDecomposition
    {
        public static void run()
        {
            int row1 = 4, col1 = 3;
            Matrix A1 = Matrix.Random(row1, col1);
            //QRDecomposition qrd1 = new QRDecomposition(A1);
            QRDecomposition qrd1 = A1.QRDecomposition;
            Console.WriteLine("Full-rank matrix = {0}", qrd1.IsFullRank);
            Console.WriteLine("Q matrix = {0}", qrd1.Q.ToString());
            Console.WriteLine("R matrix = {0}", qrd1.R.ToString());
        }
    }
}
