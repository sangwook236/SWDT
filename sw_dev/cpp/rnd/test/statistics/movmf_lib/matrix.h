// File: matrix.h -- data structures
// Author: Suvrit Sra


#ifndef _MMATRIX_H_
#define _MMATRIX_H_

#define MAX_DESC_STR_LENGTH	9

// memory for word-document matrix in CCS format
struct doc_mat
{
  char strDesc[MAX_DESC_STR_LENGTH];	// description string
  int n_row, n_col;
  int n_nz;
  int *col_ptr;
  int *row_ind;
  float *val;
};


#endif // _MMATRIX_H_
