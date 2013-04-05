/*	Sparse Matrix header file
 *		SparseMatrix.h
 *	Copyright (c) 2000, Yuqiang Guan
 */

#if !defined(_SPARSE_MATRIX_DOUBLE_H_)
#define _SPARSE_MATRIX_DOUBLE_H_

typedef float *VECTOR_double;

class SparseMatrixDouble
{
private:
	int	n_row, n_col, n_nz;

	float	*vals;
	int	*colptrs;

	// data structures for accelerating the computation of transpose multiplication
/*	int	*rowptrs;	// row pointer array
	int	*colinds;
	double  *vals2; */

public:
	int	*rowinds;

	SparseMatrixDouble(int row, int col, int nz, float *val, int *rowind, int *colptr);
	~SparseMatrixDouble();

//	void			prepare_trans_mult();

	inline float&		val(int i) { return vals[i]; }
	inline int&		row_ind(int i) { return rowinds[i]; }
	inline int&		col_ptr(int i) { return colptrs[i]; }

	inline int		GetNumRow() { return n_row; }
	inline int		GetNumCol() { return n_col; }
	inline int		GetNumNonzeros() { return n_nz; }

	float			operator() (int i, int j) const;
	void			trans_mult(float *x, float *result) ;	
};

#endif // !defined(_SPARSE_MATRIX_DOUBLEH_)



