#include <assert.h>
#include <stdlib.h>
#include "DenseMatrixDouble.h"
//#include "SparseMatrixDouble.h"

DenseMatrixDouble::DenseMatrixDouble ( int row, int col, float ** val)
{
  m_row = row;
  m_col = col;

  m_val = val;
  
  /*m_val = new VECTOR_double [m_col];
  for ( int i=0; i< m_col; i++)
    m_val[i] = new float[n_row];

  for(int i=0; i<m_row; i++)
    for (int j=0; j<m_col; j++)
    m_val[i][j]=val[i][j]; */
}

/* Does Q'A, Q' is 'm_col' by 'm_row' and A is 'a->n_row' by 'a->n_col' */  
void DenseMatrixDouble::TransMulti(SparseMatrixDouble *a, VECTOR_double *b)
{
  float * x= new float [m_row];
  float * result = new float [a->GetNumCol()];
  for ( int i=0; i<m_col; i++)
    {
      for (int j=0; j< m_row;j++)
	x[j]=m_val[j][i];

      a->trans_mult(x, result);
      
      for (int j=0; j< m_row;j++)
	b[i][j] = result [j];
    }
}

void DenseMatrixDouble::trans_mult(float *x, float *result) const
{
  for (int i = 0; i < m_col; i++)
    {
      result[i] = 0.0;
      for (int j = 0; j< m_row; j++)
	result[i] += m_val[j][i]*x[j];
    }
}

/*

// Does y = A * x, A is mxn, and y & x are mx1 and nx1 vectors
//   respectively  
void DenseMatrixDouble::dmatvec(int m, int n, float **a, float *x, float *y)
{
  float yi;
  int i,j;

  for(i = 0;i < m;i++){
    yi = 0.0;
    for(j = 0;j < n;j++){
      yi += a[i][j] * x[j];
    }
    y[i] = yi;
  }
}

// Does y = A' * x, A is mxn, and y & x are nx1 and mx1 vectors
//   respectively  
void DenseMatrixDouble::dmatvecat(int m, int n, float **a, float *x, float *y)
{
  float yi;
  int i,j;

  for(i = 0;i < n;i++){
    yi = 0.0;
    for(j = 0;j < m;j++){
      yi += a[j][i] * x[j];
    }
    y[i] = yi;
  }
}

// The procedure dqrbasis outputs an orthogonal basis spanned by the rows
//   of matrix a (using the QR Factorization of a ).  
void DenseMatrixDouble::dqrbasis( float **q)
{
  int i,j;
  float *work = new float [m_row];

  for(i = 0; i < m_row;i++){
    dmatvec( i, m_col, q, m_val[i], work );
    dmatvecat( i, m_col, q, work, q[i] );
    for(j = 0; j < m_col;j++)
      q[i][j] = m_val[i][j] - q[i][j];
    dvec_l2normalize( m_col, q[i] );
  }
}

// The function dvec_l2normsq computes the square of the
//  Euclidean length (2-norm) of the double precision vector v 
float DenseMatrixDouble::dvec_l2normsq( int dim, float *v )
{
  float length,tmp;
  int i;

  length = 0.0;
  for(i = 0;i < dim;i++){
    tmp = *v++;
    length += tmp*tmp;
  }
  return(length);
}
// The function dvec_l2normalize normalizes the double precision
//   vector v to have 2-norm equal to 1 
void DenseMatrixDouble::dvec_l2normalize( int dim, float *v )
{
  float nrm;

  nrm = sqrt(dvec_l2normsq( dim, v ));
  //if( nrm != 0 ) dvec_scale( 1.0/nrm, dim, v );
}
*/
