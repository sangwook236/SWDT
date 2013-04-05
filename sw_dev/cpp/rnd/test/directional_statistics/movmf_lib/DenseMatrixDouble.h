/*
definition of dense matrix
*/

#if !defined(_DENSE_MATRIX_DOUBLE_H_)
#define _DENSE_MATRIX_DOUBLE_H_
#include "SparseMatrixDouble.h"

class DenseMatrixDouble
{
 private:

  int m_row, m_col;
  float ** m_val;
  
 public:
  
  DenseMatrixDouble ( int row, int col, float ** val);
  void TransMulti(SparseMatrixDouble *a, VECTOR_double *b);
  void trans_mult(float *x, float *result) const;
  /*void dmatvec(int m, int n, float **a, float *x, float *y);
  void dmatvecat(int m, int n, float **a, float *x, float *y);
  void dqrbasis( float **q);
  float dvec_l2normsq( int dim, float *v );
  void dvec_l2normalize( int dim, float *v );*/
  inline int GetNumRow ()
  {
    return m_row;
  };
  inline int GetNumCol ()
    {
      return m_col;
    };
  inline int GetNumNonzeros()
    {
      return m_col*m_row;
    };
  inline float& val(int i, int j) {return m_val[i][j]; }

};

#endif // !defined(_DENSE_MATRIX_DOUBLEH_)





