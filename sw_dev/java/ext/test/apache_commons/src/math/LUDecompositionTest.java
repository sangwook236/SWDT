package math;

import org.apache.commons.math3.linear.*;

class LUDecompositionTest {

	public static void run(String[] args)
	{
		final double [][] arrMat = { { 1, 2, 3 }, { 4, 6, -2 }, { 10, 3, 1 } }; 
		final double [] arrVec = { 0, -2, 1 }; 

		final RealMatrix mat = new Array2DRowRealMatrix(arrMat);
		final RealVector vec = new ArrayRealVector(arrVec);

		LUDecomposition lud = new LUDecomposition(mat);

		final double det = lud.getDeterminant();
		final RealMatrix matL = lud.getL();
		final RealMatrix matU = lud.getU();
		DecompositionSolver	solver = lud.getSolver();

		final RealVector sol = solver.solve(vec);

		System.out.printf("%1$s%n%2$s%n%3$s%n%4$s%n", det, matL, matU, sol);
	}

}
