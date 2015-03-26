/*	mat_vec.h

	subroutines to handle sparse matrix and vector

	Copyright(c) 2000 Shulin Ni
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


#if !defined(_MAT_VEC_H_)
#define _MAT_VEC_H_

//#include "SparseMatrixDouble.h"
#include "DenseMatrixDouble.h"

// document encoding 
#define NORM_TERM_FREQ			1 // normalized term frequency
#define NORM_TERM_FREQ_INV_DOC_FREQ	2 // normalized term frequency-inverse
					  // document frequency

// normalize the norm of every column vector to 1
void normalize_mat(SparseMatrixDouble *mat);
void normalize_mat(DenseMatrixDouble *mat);

// Encode documents using the specified encoding scheme.
// The elements of the input matrix contains the number of occurrences of a word
// in a document.
void encode_mat(SparseMatrixDouble *mat, int scheme = NORM_TERM_FREQ);
void encode_mat(DenseMatrixDouble *mat, int scheme = NORM_TERM_FREQ);

// normalize the norm of vector v to 1 if it was positive
// void normalize_vec(VECTOR_double *vec);
void normalize_vec(float *vec, int n);

// compute the dot product of the col'th document vector of docs and concept_vector
float dot_mult(SparseMatrixDouble *mat, int col, float *vec);
float dot_mult(DenseMatrixDouble *mat, int col, float *vec);
// compute the dot product of two vectors
float dot_mult(float *v1, float *v2, int n);
float cosine(SparseMatrixDouble*, int, float*, int);
float cosine(float*, float*, int);

float Eucl_dist(SparseMatrixDouble *mat, int col, float *vec);

void dmatvec(int m, int n, float **a, float *x, float *y);
void dmatvecat(int m, int n, float **a, float *x, float *y);
void dqrbasis( int m, int n, float **a, float **q , float *work);
float dvec_l2normsq( int dim, float *v );
void dvec_l2normalize( int dim, float *v );
void dvec_scale( float alpha, int n, float *v );

#endif // !defined(_MAT_VEC_H_)
