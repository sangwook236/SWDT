package ext;

import org.apache.commons.math.linear.*;

public class LUDecomposition {
	public static void runAll()
	{
		final double [][] arrMat = { { 1, 2, 3 }, { 4, 6, -2 }, { 10, 3, 1 } }; 
		final double [] arrVec = { 0, -2, 1 }; 

		final RealMatrix mat = new Array2DRowRealMatrix(arrMat);
		final RealVector vec = new ArrayRealVector(arrVec);
		
		LUDecompositionImpl lud = new LUDecompositionImpl(mat);
		
		final double det = lud.getDeterminant();
		final RealMatrix matL = lud.getL();
		final RealMatrix matU = lud.getU();
		DecompositionSolver	solver = lud.getSolver();
		
		final RealVector sol = solver.solve(vec);

		System.out.printf("%1$s%n%2$s%n%3$s%n%4$s%n", det, matL, matU, sol);
	}
}
