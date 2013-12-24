using System;
using System.Collections.Generic;
using System.Text;

namespace math_library.math_net
{
    using MathNet.Numerics.LinearAlgebra;

    class Svd
    {
        public static void run()
        {
            int row1 = 4, col1 = 3;
            Matrix A1 = Matrix.Random(row1, col1);
            //SingularValueDecomposition svd1 = new SingularValueDecomposition(A1);
            SingularValueDecomposition svd1 = A1.SingularValueDecomposition;
            Console.WriteLine("Singular values = {0}", svd1.SingularValues.ToString());
            //Console.WriteLine("Singular values = {0}", svd1.S.ToString());
            Console.WriteLine("Left singular vectors = {0}", svd1.LeftSingularVectors.ToString());
            Console.WriteLine("Right singular vectors = {0}", svd1.RightSingularVectors.ToString());
        }
    }
}
