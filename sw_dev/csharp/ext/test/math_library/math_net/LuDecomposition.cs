using System;
using System.Collections.Generic;
using System.Text;

namespace math_library.math_net
{
    using MathNet.Numerics.LinearAlgebra;

    class LuDecomposition
    {
        public static void run()
        {
            int dim1 = 5;
            Matrix A1 = Matrix.Random(dim1, dim1);
            //LUDecomposition lud1 = new LUDecomposition(A1);
            LUDecomposition lud1 = A1.LUDecomposition;
            Console.WriteLine("Singular matrix = {0}", !lud1.IsNonSingular);
            Console.WriteLine("Upper triangular matrix = {0}", lud1.U.ToString());
            Console.WriteLine("Lower triangular matrix = {0}", lud1.L.ToString());
        }
    }
}
