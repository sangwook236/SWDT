/*	movmf_base.h
	Data structures used in spherical k-means algorithm
*/
/* Derived from the original movmf_base.h code written by Yuqiang Guan
 * Additions specific to movmf clustering made by:
 * Arindam Banerjee and Suvrit Sra
 * Copyright lies with the authors
 * The University of Texas at Austin 
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


#if !defined(_SPKMEANS_H_)
#define _SPKMEANS_H_


#define MAX_DESC_STR_LENGTH	9
#define MAX_VER_DIGIT_LENGTH 	3
#define SPKMEANS                200
#define QRDSPK                  201

#include "DenseMatrixDouble.h"
//#include "SparseMatrixDouble.h"
#include "RandomGenerator.h"
#include <list> 
#include <algorithm> 
#include <stdlib.h>

// initialization methods of concept vectors

#define EUCLIDEAN_K_MEANS               1
#define SPHERICAL_K_MEANS		2

#define RANDOM_PERTURB_INIT		1
#define SAMPLE_INIT			2
#define CONCEPT_VECTORS_INIT		3
#define RANDOM_INIT			4
#define FROM_FILE                       5

#define DEFAULT_EPSILON			0.001
#define DEFAULT_PERTURB			0.1
#define DEFAULT_NUM_SAMPLE		4
#define DEFAULT_SAMPLE_SIZE		0.1

#define NON_WEIGHTED_OBJ		1
#define WEIGHTED_OBJ			2

#define CD_QR                           50
#define CON_DEC                         51
#define QR_DEC                          52

class movmf_base{
  int* m_label;
  long memory_consume;
  bool silent;
 public:
  movmf_base(SparseMatrixDouble *p_docs, bool soft_assign, int num_clusters, 
	   int kappa, int init_method, char init_file[], float epsilon = DEFAULT_EPSILON, int kmax = 2);
  ~movmf_base();
  
  void Init(int cluster[]);
  
  long get_memory_consume() const { return memory_consume;}
  void set_memory_consume(long m) { memory_consume = m;}
  void incr_memory_consume(long v) { memory_consume += v;}
  // ugly, add some assertions
  void SetAlgorithm(int alg) { Alg = alg; }
  void SetEpsilon(float epsilon) { Epsilon = epsilon; }
  void SetPerturb(float perturb) { Perturb = perturb; }
  void SetInitMethod(int init_method) { Init_Method = init_method; }
  void SetConceptVectors(VECTOR_double *vec) { Concept_Vector = vec; }
  void SetNumSamples(int num_samples) { n_Samples = num_samples; }
  void SetBalance(float balance) { Balance = balance; }
  void SetEncodeScheme(int encode_scheme) { EncodeScheme = encode_scheme; }
  void SetObjFunc(int obj_func) { ObjFunc = obj_func; }
  void SetDump(bool dump_csize) { DumpCSize = dump_csize; }
  void RandSelect();
  void FullSelect();
  int GetNumIterations() { return n_Iters; }
  float GetMembership(int i, int j); //membership of point i in cluster j
  float GetClusterSize(int j) { return ClusterSize[j]; }
  float GetResult() { return Result; }
  float prior(int);
  int indexofMax(float*);

  void set_silent(bool s) { silent = s;}
  // SS: Added function to compute internal measures of similarity
  // Date: 2/24/03
  void ComputeInternalMeasures(SparseMatrixDouble*);
  void dumpsoft();

  void InitIterate(SparseMatrixDouble *p_Docs);  
  void Classify(SparseMatrixDouble* t_docs);
  int  GetClassLabel(int i);

  void Iterate(SparseMatrixDouble *p_Docs, bool soft_assign);  // iterative computation of concept vectors
  void FSIterate(SparseMatrixDouble *p_Docs);  // iterative computation of concept vectors
  void update_Sim(SparseMatrixDouble *p_Docs);

  void wordCluster(char *output_matrix, int n_Clusters);

  // sort clusters according to their sizes in non-decreasing order
  int SortCluster(int * cluster_size);		

  void DumbPopulate(SparseMatrixDouble *p_Docs); // dumb-populating clusters
  void Populate(SparseMatrixDouble *p_Docs);     // populating clusters
  void DumbSort(int p, int r);
  int Partition(int p, int r);
  
 protected:
  int Alg;			// algorithm: classic or spherical
  int *Cluster;			// which cluster a document belongs to
  int *WordCluster;
  VECTOR_double *Concept_Vector;// concept vectors

  char initFile[256];
  bool isSoft;
  float **old_CV;
  float *diff;
  float *ClusterSize;
  float *Dispersion;        // dispersion (kappa) around the mean direction
  float *logBessel;
  float *DocNormSqr;		// square of 2-norm of document vectors
  float *CVNormSqr;		// square of 2-norm of concept vectors
  
  RandomGenerator_MT19937 rand_gen;
  float order;
  int n_Words,n_Docs;	// numbers of words and documents of document matrix
  int Init_Method;		// concept vector initialization method
  int n_Clusters;		// number of clusters, equal to number of concept vectors
  int n_Iters;			// number of iterations
  float KappaBar;                 // upper bound on dispersion
  int KappaMax;                 // User given upper bound
  int n_Samples;
  int n_Pop;
  double avg_Docs;
  float Epsilon;		// stopping criteria of k-means algorithm
  float Perturb;		// maximum norm of the perturb vectors
  float Balance;
  //  float SampleSize;
  int EncodeScheme;
  int ObjFunc;
  
  // the similarity between a document and the cluster centroid (the concept vector)
  float *Inner_Sim;		
 
  float **Sim_Mat;		// similarity matrix
  float **Soft_Cluster;          // cluster assignment probabilities
  int *Samples, *Non_Samples;
  
  float Result;			// final objective function value
  
  typedef struct{
    float sim;
    int rank;
  }Sim_Info;
  
  Sim_Info *Sim_Vec;
  
  // the objective function, return the sum of dot products
  // (cosine similarity) of document vectors with their respective concept vectors
  float coherence(SparseMatrixDouble *p_Docs);		
  float soft_coherence(SparseMatrixDouble *p_Docs);		
  float full_coherence(SparseMatrixDouble *p_Docs);

  // assign document vectors to clusters according
  // to their distance to concept_vectors.
  void assign_cluster(SparseMatrixDouble *p_Docs); 
  void soft_assign_cluster(SparseMatrixDouble *p_Docs); 
  void FSassign_cluster(SparseMatrixDouble *p_Docs); 
  
  void compute_cluster_size();
  void soft_compute_cluster_size();
  
  // initialize concept vectors by perturbing the centroid
  // of document vectors, the result concept vectors are normalized
  void random_perturb_init(SparseMatrixDouble *p_Docs);	

  // initialize by reading from file
  void getInitFromFile();
  
  // totally random initialization
  void random_init(SparseMatrixDouble *p_Docs);		
  
  bool DumpCSize;
  void dump();
};

#endif // !defined(_SPKMEANS_H_)
