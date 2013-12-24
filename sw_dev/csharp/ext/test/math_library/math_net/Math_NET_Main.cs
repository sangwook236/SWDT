using System;
using System.Collections.Generic;
using System.Text;

namespace math_library
{
    class Math_NET_Main
    {
        public static void run(string[] args)
        {
            Console.WriteLine("******************* Vector Operation");
            //VectorOperation.run();  // not yet implemented
            Console.WriteLine("******************* Matrix Operation");
            MatrixOperation.run();

            Console.WriteLine("******************* LU Decomposition");
            LuDecomposition.run();
            Console.WriteLine("******************* Cholesky Decomposition");
            CholeskyDecomposition.run();
            Console.WriteLine("******************* QR Decomposition");
            QrDecomposition.run();
            Console.WriteLine("******************* Eigen-Decomposition");
            EigenDecomposition.run();
            Console.WriteLine("******************* Singular Value Decomposition");
            Svd.run();
        }
    }
}
