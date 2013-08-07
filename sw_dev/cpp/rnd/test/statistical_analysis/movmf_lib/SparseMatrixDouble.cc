/* Implementation of SparseMatrixDouble class
 * Copyright (c) 2000, Shulin Ni
 */

#include <assert.h>
#include <stdlib.h>
#include "SparseMatrixDouble.h"

/*VECTOR_double::VECTOR_double(int s)
{
	size = s;
	val = new double[size];
}*/

SparseMatrixDouble::SparseMatrixDouble(int row, int col, int nz, float *val, int *rowind, int *colptr)
{
  //int i;

	n_row = row;
	n_col = col;
	n_nz = nz;

	/*vals = new double[n_nz];
	rowinds = new int[n_nz];
	colptrs = new int[n_col+1];

	for (i = 0; i < n_nz; i++)
	{
		vals[i] = val[i];
		rowinds[i] = rowind[i];
	}
	for (i = 0; i < n_col+1; i++)
	colptrs[i] = colptr[i];*/
	vals = val;
	rowinds = rowind;
	colptrs = colptr;
}

/* void SparseMatrixDouble::prepare_trans_mult()
{
	int i, j;

	// setting up data structures for transpose multiplication
	int *rp = new int[n_row];
	rowptrs = new int[n_row+1];
	colinds = new int[n_nz];
	vals2 = new double[n_nz];
	
	for (i = 0; i < n_row+1; i++)
		rowptrs[i] = 0;
	for (i = 0; i < n_nz; i++)
		rowptrs[rowinds[i]+1]++;
	for (i = 1; i < n_row+1; i++)
		rowptrs[i] += rowptrs[i-1];
	
	assert(rowptrs[n_row] == n_nz && colptrs[n_col] ==n_nz);

	for (i = 0; i < n_row; i++)
		rp[i] = rowptrs[i];
	for (i = 0; i < n_col; i++)
		for (j = colptrs[i]; j < colptrs[i+1]; j++)
		{
			colinds[rp[rowinds[j]]] = i;
			vals2[rp[rowinds[j]]] = vals[j];
			rp[rowinds[j]]++;
		}
}*/

SparseMatrixDouble::~SparseMatrixDouble()
{
  /*delete[] vals;
	delete[] rowinds;
	delete[] colptrs;*/

	// ... delete data structures for transpose multiplication
/*	delete[] rowptrs;
	delete[] colinds;
	delete[] vals2; */
}

float SparseMatrixDouble::operator()(int i, int j) const
{
	assert(i >= 0 && i < n_row && j >= 0 && j < n_col);

	for (int t = colptrs[j]; t < colptrs[j+1]; j++)
		if (rowinds[t] == i) return vals[t];
	return 0.0;
}

void SparseMatrixDouble::trans_mult(float *x, float *result) 
{
	for (int i = 0; i < n_col; i++)
	{
	  result[i] = 0.0;
	  for (int j = colptrs[i]; j < colptrs[i+1]; j++)
	    result[i] += vals[j] * x[rowinds[j]];
	}
	
}



