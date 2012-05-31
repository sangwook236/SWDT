using System;
using System.Collections.Generic;
using System.Text;

namespace mathnet_test
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("******************* Vector Operation");
                VectorOperation.run();  // not yet implemented
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
            catch (Exception e)
            {
                Console.WriteLine("System.Exception occurred: {0}", e);
            }

            Console.WriteLine("press any key to exit ...");
            Console.ReadKey();
        }
    }
}
