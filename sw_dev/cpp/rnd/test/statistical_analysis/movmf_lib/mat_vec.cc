/*	mat_vec.cc

	Implementation of the Sparse_Mat_double class

	Copyright(c) 2000 Yuqiang Guan, Shulin Li
    Copyright The University of Texas at Austin 
*/
/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */

/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */

/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA. */

#include <iostream>
#include <cmath>
#include "mat_vec.h"

/* The procedure dqrbasis outputs an orthogonal basis spanned by the rows
   of matrix a (using the QR Factorization of a ).  */
void dqrbasis( int m, int n, float **a, float **q , float *work)
{
  int i,j;

  for(i = 0; i < m;i++){
    dmatvec( i, n, q, a[i], work );
    dmatvecat( i, n, q, work, q[i] );
    for(j = 0; j < n;j++)
      q[i][j] = a[i][j] - q[i][j];
    dvec_l2normalize( n, q[i] );
  }
}

/* Does y = A * x, A is mxn, and y & x are mx1 and nx1 vectors
   respectively  */
void dmatvec(int m, int n, float **a, float *x, float *y)
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

/* Does y = A' * x, A is mxn, and y & x are nx1 and mx1 vectors
   respectively  */
void dmatvecat(int m, int n, float **a, float *x, float *y)
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

/* The function dvec_l2normsq computes the square of the
   Euclidean length (2-norm) of the double precision vector v */
float dvec_l2normsq( int dim, float *v )
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

/* The function dvec_l2normalize normalizes the double precision
   vector v to have 2-norm equal to 1 */
void dvec_l2normalize( int dim, float *v )
{
  float nrm;

  nrm = sqrt(dvec_l2normsq( dim, v ));
  if( nrm != 0 ) dvec_scale( 1.0/nrm, dim, v );
}

void dvec_scale( float alpha, int n, float *v )
{
  int i;

  for(i = 0;i < n;i++){
    *v++ = *v * alpha;
  }
}

void normalize_mat(SparseMatrixDouble *mat)
{
	int i, j;
	float norm;

	for (i = 0; i < mat->GetNumCol(); i++)
	{
		norm = 0.0;
		for (j = mat->col_ptr(i); j < mat->col_ptr(i+1); j++)
		norm += mat->val(j) * mat->val(j);
//              assert(norm > 0);
		/*		if (norm == 0)
		{
			cerr << "column " << i << " has 0 norm\n";
			}*/
		if(norm!=0)
		{
		  norm = sqrt(norm);
		  for (j = mat->col_ptr(i); j < mat->col_ptr(i+1); j++)
		    mat->val(j) /= norm;
	        }
	}
}

void normalize_mat(DenseMatrixDouble *mat)
{
  int i, j;
  float norm;

  for ( i=0; i < mat->GetNumCol(); i++)
    {
      norm = 0.0;
      for (j=0; j < mat->GetNumRow(); j++)
	norm+=mat->val(j, i)*mat->val(j, i);
    
      if(norm!=0)
	{
	  norm = sqrt(norm);
	  for (j=0; j<mat->GetNumRow() ; j++)
	    mat->val(j, i) /= norm;
	}
    }
}

void encode_mat(SparseMatrixDouble *mat, int scheme)
{
	int i;
	int *n_doc_per_word;	// number of documents which contains a word
	float *global_term;
	
	switch (scheme)
	{
	case NORM_TERM_FREQ:
		normalize_mat(mat);
		break;
	case NORM_TERM_FREQ_INV_DOC_FREQ:
		n_doc_per_word = new int[mat->GetNumRow()];
		global_term = new float[mat->GetNumRow()];

		for (i = 0; i < mat->GetNumRow(); i++);
			n_doc_per_word[i] = 0;
		for (i = 0; i < mat->GetNumNonzeros(); i++)
			if (mat->row_ind(i) != 0)
				n_doc_per_word[mat->row_ind(i)]++;
		for (i = 0; i < mat->GetNumRow(); i++);
		{
			if (n_doc_per_word[i] == 0)
				global_term[i] = 0;
			else
				global_term[i]
					= log((float)mat->GetNumCol()/n_doc_per_word[i]);
		}
		for (i = 0; i < mat->GetNumNonzeros(); i++)
			mat->val(i) *= mat->val(i) * global_term[mat->row_ind(i)];
		normalize_mat(mat);

		break;
	default:
		break;
	}
}
void encode_mat(DenseMatrixDouble *mat, int scheme)
{
	int i, j;
	int *n_doc_per_word;	// number of documents which contains a word
	float *global_term;
	
	switch (scheme)
	{
	case NORM_TERM_FREQ:
		normalize_mat(mat);
		break;
	case NORM_TERM_FREQ_INV_DOC_FREQ:
		n_doc_per_word = new int[mat->GetNumRow()];
		global_term = new float[mat->GetNumRow()];

		for (i = 0; i < mat->GetNumRow(); i++);
			n_doc_per_word[i] = 0;
		for (i = 0; i < mat->GetNumRow(); i++)
		  for(j=0; j<mat->GetNumCol(); j++)
			if (mat->val(i, j) != 0)
				n_doc_per_word[i]++;
		for (i = 0; i < mat->GetNumRow(); i++);
		{
			if (n_doc_per_word[i] == 0)
				global_term[i] = 0;
			else
				global_term[i]
					= log((float)mat->GetNumCol()/n_doc_per_word[i]);
		}
		for (i = 0; i < mat->GetNumRow(); i++)
		  for(j=0; j<mat->GetNumCol(); j++)
	 	      mat->val(i, j) *= mat->val(i, j) * global_term[i];
		normalize_mat(mat);

		break;
	default:
		break;
	}
}


// normalize the norm of vector v to 1 if it was positive
/*void normalize_vec(float *vec, int size)
{
	float norm;
	int i;

	norm = 0.0;
	for (i = 0; i < size; i++)
	norm += vec[i] * vec[i];

//      assert(norm > 0);
	if (norm == 0) return;
	norm = sqrt(norm);

	for (i = 0; i < size; i++)
	vec[i] /= norm;
}*/

void normalize_vec(float vec[], int n)
{
        float norm;
        int i;

        norm = 0.0;
        for (i = 0; i < n; i++)
        norm += vec[i] * vec[i];

//      assert(norm > 0);
        if (norm == 0) return;
        norm = sqrt(norm);

        for (i = 0; i < n; i++)
        vec[i] /= norm;
}

// compute the dot product of the col'th document vector of docs and concept_vector
float dot_mult(SparseMatrixDouble *mat, int col, float *vec)
{
	float result = 0.0;
	int j;

	for (j = mat->col_ptr(col); j < mat->col_ptr(col+1); j++)
		result += mat->val(j) * vec[mat->row_ind(j)];

	return result;
}


float Eucl_dist(SparseMatrixDouble *mat, int col, float *vec)
{
  float result = 0.0;
  int j;
  
  for (j = mat->col_ptr(col); j < mat->col_ptr(col+1); j++)
    result += (mat->val(j) - vec[mat->row_ind(j)])*(mat->val(j) - vec[mat->row_ind(j)]);
  
  
  return result;
}


float dot_mult(DenseMatrixDouble *mat, int col, float *vec)
{
  float result = 0.0;
  int j;
  
  for(j=0; j<mat->GetNumRow(); j++)
      result+=mat->val(j, col)*vec[j];
  return result;
}

float dot_mult(float *v1, float *v2, int n)
{
	float result = 0.0;
	int j;

	for (j = 0; j < n; j++)
		result += v1[j] * v2[j];

	return result;
}


// x^ty/\no{x}\no{y}
float cosine(SparseMatrixDouble* mat, int col, float* vec, int lv)
{
  float normy = dvec_l2normsq(lv, vec);
  float normx = 0.0;
  float result = 0.0;
  float tmp;

  for (int j = mat->col_ptr(col); j < mat->col_ptr(col+1); j++) {
    tmp = mat->val(j);
    result += tmp * vec[mat->row_ind(j)];
    normx += tmp*tmp;
  }

  float prod = sqrt(normx * normy);
  if (prod == 0)
    return 0.0;
  return result/prod;
}


float cosine(float* v1, float* v2, int n)
{
  float result = 0.0;

  float n1 = 0, n2 = 0;
  for (int j = 0; j < n; j++) {
    result += v1[j]*v2[j];
    n1 += v1[j]*v1[j];
    n2 += v2[j]*v2[j];
  }
  float prod = sqrt(n1*n2);

  if (prod == 0) return 0;
  return result/prod;
}

