using System;
using System.Collections.Generic;
using System.Text;

namespace math_library.math_net
{
    using MathNet.Numerics.LinearAlgebra;

    class EigenDecomposition
    {
        public static void run()
        {
            int dim1 = 5;
            Matrix A1 = Matrix.Random(dim1, dim1);
            //EigenvalueDecomposition evd1 = new EigenvalueDecomposition(A1);
            EigenvalueDecomposition evd1 = A1.EigenvalueDecomposition;
            Console.WriteLine("Eigenvalues = {0}", evd1.EigenValues.ToString());
            Console.WriteLine("Eigenvectors = {0}", evd1.EigenVectors.ToString());
        }
    }
}
