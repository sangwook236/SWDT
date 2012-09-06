using System;
using System.Collections.Generic;
using System.Text;

namespace mathnet
{
    using MathNet.Numerics.LinearAlgebra;

    class CholeskyDecomposition
    {
        public static void run()
        {
            int dim1 = 5;
            Matrix A1 = Matrix.Random(dim1, dim1);
            A1.Add(Matrix.Transpose(A1));

            //MathNet.Numerics.LinearAlgebra.CholeskyDecomposition chol1 = new MathNet.Numerics.LinearAlgebra.CholeskyDecomposition(A1);
            MathNet.Numerics.LinearAlgebra.CholeskyDecomposition chol1 = A1.CholeskyDecomposition;
            Console.WriteLine("symmetric and positive definite matrix = {0}", chol1.IsSPD);
            if (chol1.IsSPD)
                Console.WriteLine("Triangular factor matrix  = {0}", chol1.TriangularFactor.ToString());

            int dim2 = 5;
            double[] data2 = { 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 3, 6, 10, 15, 1, 4, 10, 20, 35, 1, 5, 15, 35, 70 };
            Matrix A2 = new Matrix(data2, dim2);

            //MathNet.Numerics.LinearAlgebra.CholeskyDecomposition chol2 = new MathNet.Numerics.LinearAlgebra.CholeskyDecomposition(A2);
            MathNet.Numerics.LinearAlgebra.CholeskyDecomposition chol2 = A2.CholeskyDecomposition;
            Console.WriteLine("symmetric and positive definite matrix = {0}", chol2.IsSPD);
            if (chol2.IsSPD)
                Console.WriteLine("Triangular factor matrix  = {0}", chol2.TriangularFactor.ToString());
        }
    }
}
