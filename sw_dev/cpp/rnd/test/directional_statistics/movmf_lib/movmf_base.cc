/*	movmf_base.cc 

Implementation of the movmf_base class
*/
/* Derived from the original SPKMeans.cc code written by Yuqiang Guan
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


#include <time.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "mat_vec.h"
#include "movmf_base.h"
#include "newbessel.h"

using namespace std;

#define PI to_RR("3.141592653589793")
#define fPI to_float(PI)

//#define CV_Sim(i, j) CV_Sim[(i>=j)?(i*(i+1)/2+j):(j*(j+1)/2+i)]

movmf_base::movmf_base(SparseMatrixDouble *p_Docs, 
                       bool soft_assign, 
                       int num_clusters,
                       int kappa, int init_method, char init_file[],
                       float epsilon, int kmax)
{
  assert(p_Docs != NULL && num_clusters > 0 &&  epsilon > 0);

  memory_consume = 0;
  //pDocs = p_docs;
  // change this according to soft_assign criterio
  // will have to pass soft_assign to movmf_base
  // then cluster will be int [] or float [][]
  n_Clusters = num_clusters;
  Init_Method = init_method;
  if(Init_Method == FROM_FILE){
    strcpy(initFile,init_file);
  }
  Epsilon = epsilon;
  isSoft = soft_assign;
  n_Words =  p_Docs->GetNumRow();
  order = n_Words;
  order /= 2.0;
  n_Docs = p_Docs->GetNumCol();
  n_Iters = 0;
  Result = 0;
  Alg = SPHERICAL_K_MEANS;
  Perturb = DEFAULT_PERTURB;
  n_Samples = DEFAULT_NUM_SAMPLE;
  Balance = 1.0;
  EncodeScheme = NORM_TERM_FREQ;
  ObjFunc = NON_WEIGHTED_OBJ;
  DumpCSize = false;
  KappaBar = kappa;
  KappaMax = kmax;
  Dispersion = new float[n_Clusters];
  logBessel = new float[n_Clusters];
  Inner_Sim = new float[n_Docs];
  Sim_Mat = new float *[n_Clusters];
  for (int j = 0; j < n_Clusters; j++)
    Sim_Mat[j] = new float[n_Docs];
     
  if(soft_assign){
    Soft_Cluster = new float *[n_Docs];
    for (int j = 0; j < n_Docs; j++)
      Soft_Cluster[j] = new float[n_Clusters];
	  
    Cluster = new int[n_Docs];
  }
     
     
  memory_consume+=(n_Docs+n_Clusters*n_Docs/2)*sizeof(float) + n_Clusters*sizeof(int);

  if (init_method != CONCEPT_VECTORS_INIT)
    {
      Concept_Vector = new VECTOR_double[n_Clusters];
      for (int i = 0; i < n_Clusters; i++)
        Concept_Vector[i] = new float[n_Words];
    }

  //	qTransA_Vector = cTransA_Vector = NULL;

  memory_consume+=n_Clusters*n_Words*sizeof(float)*2;

  old_CV = new VECTOR_double[n_Clusters];
  for (int i = 0; i < n_Clusters; i++)
    old_CV[i] = new float[n_Words];
  diff = new float[n_Clusters];
  ClusterSize = new  float[n_Clusters];
  DocNormSqr = new float[n_Docs];
  CVNormSqr = new float[n_Clusters];

  memory_consume+=(n_Clusters*2+n_Docs+n_Clusters)*sizeof(float);
  //rand_gen.Set((unsigned)0);

  rand_gen.Set((unsigned)time(NULL));
}


void movmf_base::Init(int cluster[]){
  assert(cluster != NULL);
  Cluster = cluster;
}

// Classify the docs in t_docs -- uses the cluster info in p_docs below
void movmf_base::Classify(SparseMatrixDouble *t_docs) {
  m_label = new int[t_docs->GetNumCol()];
  int ncol = t_docs->GetNumCol();
  for (int i = 0; i < ncol; i++) {
    m_label[i] = 0;
  }

  int i, j;
  float** sval = new float*[ncol];
  for (i = 0; i < ncol; i++)
    sval[i] = new float[n_Clusters];
  

  float* localSim = new float[ncol];
  // Perform classification
  for (i = 0; i < ncol; i++) {
    for (j = 0; j < n_Clusters; j++) {
      t_docs->trans_mult(Concept_Vector[j], localSim);
      sval[i][j] =  Dispersion[j]*localSim[i] + log(prior(j));
    }
    m_label[i] = indexofMax(sval[i]);
  }

  for (i = 0; i < ncol; i++)
    delete[] sval[i];
  delete[] sval;
}

float movmf_base::prior(int j) {
  return ClusterSize[j]/n_Samples;
}

int movmf_base::indexofMax(float* s)
{
  int r = 0;
  float cmax = -1;

  for (int i = 0; i < n_Clusters; i++) {
    if (s[i] > cmax) {
      cmax = s[i];
      r = i;
    }
  }
  return r;
}
// Return class label of document i, basically col i of t_docs
int movmf_base::GetClassLabel(int i) {
  return m_label[i];
}

void movmf_base::InitIterate(SparseMatrixDouble *p_Docs){
  float temp_result;
  int i, j;
  clock_t start_clock, finish_clock;

  // normalize document matrix if using spherical k means algorithm
  // else compute norms of document vectors
  if (Alg == SPHERICAL_K_MEANS){
    normalize_mat(p_Docs);
  }
  else{
    for (i = 0; i < n_Samples; i++){
      DocNormSqr[i] = 0;
      for (j = p_Docs->col_ptr(Samples[i]); j < p_Docs->col_ptr(Samples[i]+1); j++){
        DocNormSqr[i] += p_Docs->val(j) * p_Docs->val(j);
      }
    }
  }


  switch (Init_Method) {

  case FROM_FILE:
    getInitFromFile();
    break;
  case SAMPLE_INIT:
    random_perturb_init(p_Docs);
    //refined_partition(p_Docs,n_Samples);
    break;
  case RANDOM_PERTURB_INIT:
    random_perturb_init(p_Docs);
    break;
  case RANDOM_INIT:
    random_init(p_Docs);
    if (Alg == EUCLIDEAN_K_MEANS){
      for (i = 0; i < n_Clusters; i++){
        CVNormSqr[i] = 0;
        for (j = 0; j < n_Words; j++){
          CVNormSqr[i] += Concept_Vector[i][j] * Concept_Vector[i][j];
        }
      }
    }
    break;
  default:
    break;
  }
	
	
  // initial assignment, all documents are assigned to clusters randomly !
	
  for (i = 0; i < n_Samples; i++)	   Cluster[Samples[i]] = 0;
  for (i = 0; i < n_Clusters; i++)   p_Docs->trans_mult(Concept_Vector[i], Sim_Mat[i]);
  for (i = 0; i < n_Clusters; i++)   diff[i] = 0.0;


  Result = 0.0;
  n_Iters = 0;

  start_clock = clock();

  do{
    assign_cluster(p_Docs);
    compute_cluster_size();
	  
    temp_result = Result;
    n_Iters++;
	  
	  
    // save and rsest old concept vectors
    for (i = 0; i < n_Clusters; i++)
      ////////////////////////////////////////////////////////////
      if(ClusterSize[i] != 0)
        //////////////////////////////////////////////////////////
        for (j = 0; j < n_Words; j++){
          old_CV[i][j] = Concept_Vector[i][j];
          Concept_Vector[i][j] = 0.0;
        }
	  
    // compute new concept vectors
    for (i = 0; i < n_Samples; i++){
      assert(Cluster[Samples[i]] >=0 && Cluster[Samples[i]] < n_Clusters);
      for (j = p_Docs->col_ptr(Samples[i]); j < p_Docs->col_ptr(Samples[i]+1); j++){
        Concept_Vector[Cluster[Samples[i]]][p_Docs->row_ind(j)] += p_Docs->val(j);
      }
    }
	  
    if (Alg == SPHERICAL_K_MEANS)
      for (i = 0; i < n_Clusters; i++)
        normalize_vec(Concept_Vector[i], n_Words);
    else{
      for (i = 0; i < n_Clusters; i++){
        CVNormSqr[i] = 0;
        if (ClusterSize[i] != 0)
          for (j = 0; j < n_Words; j++){
            Concept_Vector[i][j] /= ClusterSize[i];
            CVNormSqr[i] += Concept_Vector[i][j] * Concept_Vector[i][j];
          }
      }
    }
	  
    for (i = 0; i < n_Clusters; i++){
      p_Docs->trans_mult(Concept_Vector[i], Sim_Mat[i]);
    }	  
	  
    // compute the difference between old and new concept vectors
    for (i = 0; i < n_Clusters; i++){
      diff[i] = 0.0;
      for (j = 0; j < n_Words; j++)
        diff[i] += (old_CV[i][j] - Concept_Vector[i][j]) * (old_CV[i][j] - Concept_Vector[i][j]);
      diff[i] = sqrt(diff[i]);
    }
	  
    // compute new objective function
    Result = coherence(p_Docs);

    if (!silent)
      cout << Result << endl;
    //	  dump();
	  
  } while ((fabs(temp_result - Result) > fabs(Epsilon*Result)));
  finish_clock = clock();
  //	cout<<"Core kmeans (routine) time: "<<(finish_clock - start_clock)/1e6<<endl;
  //	dump();
}



void movmf_base::Iterate(SparseMatrixDouble *p_Docs, bool soft_assign)
{
  float temp_result,R_Bar=1.0,newKappa;
  int i, j, h;
  bool BoundHit = false;

  // normalize document matrix if using spherical k means algorithm
  normalize_mat(p_Docs);

  /*
    switch (Init_Method) {
    case SAMPLE_INIT:
    random_perturb_init(p_Docs);
    //refined_partition(p_Docs,n_Samples);
    break;
    case RANDOM_PERTURB_INIT:
    random_perturb_init(p_Docs);
    break;
    case RANDOM_INIT:
    random_init(p_Docs);
    if (Alg == EUCLIDEAN_K_MEANS){
    for (i = 0; i < n_Clusters; i++){
    CVNormSqr[i] = 0;
    for (j = 0; j < n_Words; j++){
    CVNormSqr[i] += Concept_Vector[i][j] * Concept_Vector[i][j];
    }
    }
    }
    break;
    default:
    break;
    }
  */	

  // initial assignment, all documents are assigned to clusters randomly !
	
  if(!soft_assign){
    for (i = 0; i < n_Samples; i++)	 
      Cluster[Samples[i]] = 0;
  }

  for (i = 0; i < n_Clusters; i++){
    p_Docs->trans_mult(Concept_Vector[i], Sim_Mat[i]);
    Dispersion[i] = KappaBar;            // initializing kappa
    diff[i] = 0.0;
  }

  Result = 0.0;
  n_Iters = 0;

  ////////////////////////////////

  // have to handle stuff here..........................
  while (KappaBar < KappaMax) {

    for (i = 0; i < n_Clusters; i++)    Dispersion[i] = KappaBar;      
	  
    do {
      if (soft_assign) {
        soft_assign_cluster(p_Docs);
        soft_compute_cluster_size();
        // dumpsoft();
      } else {
        assign_cluster(p_Docs);
        compute_cluster_size();
      }
	    
      temp_result = Result;
      n_Iters++;
	    
	    
      // save and rsest old concept vectors
      for (i = 0; i < n_Clusters; i++)	    
        if(ClusterSize[i] != 0.0)              //  a useful hack 
          for (j = 0; j < n_Words; j++){
            old_CV[i][j] = Concept_Vector[i][j];
            Concept_Vector[i][j] = 0.0;
          }
	    

      //-----------------------------------------------------------------------
      // The main changes for the moVMF case are in the following lines
      //-----------------------------------------------------------------------
	       
      // Compute NEW Concept Vectors
	  
	       
      if (soft_assign) {                // the SOFT assignment case /////////////
        for (i = 0; i < n_Samples; i++) {
          for (j = p_Docs->col_ptr(Samples[i]); j < p_Docs->col_ptr(Samples[i]+1); j++){
            for (h = 0; h < n_Clusters; h++) {		    
              Concept_Vector[h][p_Docs->row_ind(j)] += exp(Soft_Cluster[i][h])*p_Docs->val(j);
            }
          }
        }
      } else {                    	  // the HARD assignment case ///////////
        for (i = 0; i < n_Samples; i++) {
          for (j = p_Docs->col_ptr(Samples[i]); j < p_Docs->col_ptr(Samples[i]+1); j++){
            Concept_Vector[Cluster[Samples[i]]][p_Docs->row_ind(j)] += p_Docs->val(j);
          }
        }
      }
	  
	  
      for (i = 0; i < n_Clusters; i++) {
        CVNormSqr[i] = 0;
        for (j = 0; j < n_Words; j++) {
          CVNormSqr[i] += Concept_Vector[i][j]*Concept_Vector[i][j];
        }
        if (ClusterSize[i] != 0.0)
          R_Bar = sqrt(CVNormSqr[i])/ClusterSize[i];
        ///////////////////////////////////////////////////////////////////////////////////////
        newKappa = (R_Bar*n_Words - R_Bar*R_Bar*R_Bar)/(1 - R_Bar*R_Bar); // approximation
        if(newKappa < Dispersion[i]){
          Dispersion[i] = newKappa;
        }
        //	    Dispersion[i] = find_quotient_arg(order, R_Bar, 1.0);
        ////////////////////////////////////////////////////////////////////////////////////////
        if (!silent)
          cout << " Cluster " << i << " :  Size = " << ClusterSize[i] 
               << ", R_Bar = "<< R_Bar << ", kappa = " << Dispersion[i] << endl;
        normalize_vec(Concept_Vector[i], n_Words);
      }
	  
	  
      for (i = 0; i < n_Clusters; i++) {
        p_Docs->trans_mult(Concept_Vector[i], Sim_Mat[i]);
      }	  
	  
      // compute the difference between old and new concept vectors
      for (i = 0; i < n_Clusters; i++){
        diff[i] = 0.0;
        for (j = 0; j < n_Words; j++)
          diff[i] += (old_CV[i][j] - Concept_Vector[i][j]) * (old_CV[i][j] - Concept_Vector[i][j]);
        diff[i] = sqrt(diff[i]);
      }
	  
      // compute new objective function
      if(soft_assign) {
        Result = soft_coherence(p_Docs);
        //dumpsoft();
      } else { 
        Result = coherence(p_Docs);
      }
      if (!silent)
        cout << Result << endl;

    } while ((fabs(temp_result - Result) > fabs(Epsilon*Result)));
    
    // dump();			// dump values
    // cout << KappaBar << ": "; 
    for (i = 0; i < n_Clusters; i++){
      // cout << Dispersion[i] << " ; ";
      if(Dispersion[i] == KappaBar){
        BoundHit = true;
      }
    }
    // cout << endl;
	  
    if(BoundHit){
      KappaBar *= 1.412;
    }
    else{
      KappaBar = 10000;
    }
	  
  }
}


void movmf_base::soft_assign_cluster(SparseMatrixDouble *p_Docs){
  int i, j;
  static bool first = true;

  RR::SetPrecision(300);
  RR x,nu;
  RR ri, rip, rk, rkp;
  float logSum;
  RR FullSum,Temp;

  if (first) {
    nu = order-1;
    for (j = 0; j < n_Clusters; j++){
      x  = Dispersion[j];
      ri = BesselI(nu, x);
      //bessik(x, nu, ri, rk, rip, rkp);
      logBessel[j] = to_float(log(ri));
    }
    first = false;
  }

  // reassign documents
  for (i = 0; i < n_Samples; i++) {
    FullSum = to_RR(0.0);
    for (j = 0; j < n_Clusters; j++) {
      Soft_Cluster[Samples[i]][j] =  Dispersion[j]*Sim_Mat[j][Samples[i]]
        + (order-1)*log(Dispersion[j]) - (order)*log(2*fPI) - logBessel[j];
      Temp = to_RR(Soft_Cluster[Samples[i]][j]);
      FullSum += exp(Temp);
    }
    
    logSum = to_float(log(FullSum));
    for (j = 0; j < n_Clusters; j++){
      Soft_Cluster[Samples[i]][j] -= logSum;
    }
  }
}

// This dumps log's of the probabilities.
void movmf_base::dumpsoft()
{
  for (int i = 0; i < n_Samples; i++) {
    for (int j = 0; j < n_Clusters; j++)
      cout << exp(Soft_Cluster[Samples[i]][j]) << " ";
    cout << endl;
  }

}

float movmf_base::GetMembership(int i, int j){ 
  //  cout << "(" << i << " , " << j << ")" << Soft_Cluster[i][j] << endl;
  return exp(Soft_Cluster[Samples[i]][j]); 
}



void movmf_base::soft_compute_cluster_size(){
  int i,j;

  for (i = 0; i < n_Clusters; i++)  ClusterSize[i] = 0.0;

  for (i = 0; i < n_Samples; i++)
    for (j=0; j < n_Clusters; j++)
      ClusterSize[j] += exp(Soft_Cluster[Samples[i]][j]);
}



void movmf_base::assign_cluster(SparseMatrixDouble *p_Docs){
  int i, j;
  // TODO: sanity check for hard clustering

  // reassign documents
  for (i = 0; i < n_Samples; i++){
    for (j = 0; j < n_Clusters; j++){
      //      cout << "Sim value " << i << " with cluster " << j << " : " << Sim_Mat[j][Samples[i]] << endl;
      if (j != Cluster[Samples[i]]){
        if (  Dispersion[j]*Sim_Mat[j][Samples[i]] >
              //+ (order-1)*log(Dispersion[j]) > 
              Dispersion[Cluster[Samples[i]]]*Sim_Mat[Cluster[Samples[i]]][Samples[i]] ){
          //	      + (order-1)*log(Dispersion[j])      ){
          Cluster[Samples[i]] = j;
        }
      }
    }
  }
}

void movmf_base::compute_cluster_size(){
  int i;

  for (i = 0; i < n_Clusters; i++)  ClusterSize[i] = 0;
  for (i = 0; i < n_Samples; i++)   ClusterSize[Cluster[Samples[i]]]++;
}




///////////////////////////////////////////////////////////////////////////////////////////


movmf_base::~movmf_base()
{
  if (Concept_Vector != NULL && Init_Method != CONCEPT_VECTORS_INIT)
    {
      for (int i = 0; i < n_Clusters; i++)
        {
          delete[] Concept_Vector[i];
          delete[] Sim_Mat[i];
        }
      
      delete[] Concept_Vector;
      delete[] Sim_Mat;
    }
  
  delete[] ClusterSize; // any problem? ...
  delete[] DocNormSqr;
  delete[] CVNormSqr;
  delete[] Inner_Sim;
  //delete[] CV_Sim;
}


void movmf_base::dump()
{
  int i;

  for (i = 0; i < n_Clusters; i++)
    cout << "Cluster " << i << ": " << ClusterSize[i] << "\n";
  cout << "objective function value :" << GetResult() << "\n";
  // Suvrit:
  // Also dump the learned means and kappas till now at this point.
  cout << "** START KAPPAS **" << endl;
  for (i = 0; i < n_Clusters; i++) {
    cout << i << " " << Dispersion[i] << endl;
  }
  cout << "** END KAPPAS **" << endl;
  for (i = 0; i < n_Clusters; i++) {
    // print ith concept vector
    for (int j = 0; j < n_Words; j++)
      cout << Concept_Vector[i][j] << " ";
    cout << endl;
  }
}



float movmf_base::soft_coherence(SparseMatrixDouble *p_Docs){
  int i,j;
  float value = 0.0;
  RR::SetPrecision(300);
  RR x,nu;
  RR ri, rip, rk, rkp;

  nu = order-1;
  for(j = 0; j < n_Clusters; j++){
    x  = Dispersion[j];
    ri = BesselI(nu,x);
    //bessik(x, nu, ri, rk, rip, rkp);
    logBessel[j] = to_float(log(ri));
    std::cout << "LogBesselI = " << logBessel[j] << "\n";
  }
  
  for (i = 0; i < n_Samples; i++){
    for ( j = 0; j < n_Clusters; j++){ 
      value += exp( Soft_Cluster[i][j] )*( Dispersion[j]*dot_mult(p_Docs, Samples[i],Concept_Vector[j]) + (order-1)*log(Dispersion[j]) - (order)*log(2*fPI) - logBessel[j] );
    }
  }
  
  return value;
}


float movmf_base::coherence(SparseMatrixDouble *p_Docs){
  int i;
  float value = 0.0;
  
  for (i = 0; i < n_Samples; i++)	{
    value += Dispersion[Cluster[Samples[i]]]*dot_mult(p_Docs, Samples[i],Concept_Vector[Cluster[Samples[i]]]) + (order-1)*log(Dispersion[Cluster[Samples[i]]]);
  }

  return value;
}





int movmf_base::SortCluster(int *cluster_size )
{
  //int *cluster_size = new int[n_Clusters];
  int *cluster_label = new int[n_Clusters];
  int *new_label = new int[n_Clusters];
  int i, j, tmp_size, tmp_label;
  bool flag;

  for (i = 0; i < n_Clusters; i++)
    {
      cluster_size[i] = 0;
      cluster_label[i] = i;
    }
  for (i = 0; i < n_Samples; i++)
    cluster_size[Cluster[Samples[i]]]++;

  // ugly sorting
  for (i = 0; i < n_Clusters-1; i++)
    {
      flag = true;
      for (j = 0; j < n_Clusters-i-1; j++)
        if (cluster_size[j+1] < cluster_size[j])
          {
            flag = false;
            tmp_size = cluster_size[j+1];
            tmp_label = cluster_label[j+1];
            cluster_size[j+1] = cluster_size[j];
            cluster_label[j+1] = cluster_label[j];
            cluster_size[j] = tmp_size;
            cluster_label[j] = tmp_label;
          }
      if (flag) break;
    }

  // eliminate the empty clusters
  int emptyCluster=0;
  i=0;
  while(cluster_size [i++] ==0)
    emptyCluster++;
	    
  WordCluster = new int[n_Words];
  int index;
  float max;

  for (j = 0; j < n_Words; j++)
    {
      max = Concept_Vector[0][j];
      index =0;

      for (i = 1; i < n_Clusters; i++)
        if (Concept_Vector[i][j] > max) 
          {
            max = Concept_Vector[i][j];
            index =i;
          }
      WordCluster[j]=index;
    }
	
  // modify Cluster array, also ugly
  for (i = 0; i < n_Clusters; i++)
    new_label[cluster_label[i]] = i;

  for (i = 0; i < n_Samples; i++)
    Cluster[Samples[i]] = new_label[Cluster[Samples[i]]]-emptyCluster;

  for (i = 0; i < n_Words; i++)
    WordCluster[i] = new_label[WordCluster[i]]-emptyCluster;

  return emptyCluster;
}

void movmf_base::wordCluster(char *output_matrix, int n_Clusters)
{
  char browserfilepost[128];
  //double max; 
  int j;

  sprintf(browserfilepost, "_wordtoclus.%d", n_Clusters);
  strcat(output_matrix, browserfilepost);
  std::ofstream o_m(output_matrix);
  
  o_m<< n_Words <<endl;

  for (j = 0; j < n_Words; j++)
    o_m<<WordCluster[j]<<endl;
 
  o_m.close();
    
}



float movmf_base::full_coherence(SparseMatrixDouble *p_Docs)
{
  int i;
  float value = 0.0;
	
  value = coherence(p_Docs);

  if (Alg == SPHERICAL_K_MEANS){
    for (i = 0; i < n_Pop; i++)	{
      value += dot_mult(p_Docs, Non_Samples[i], Concept_Vector[Cluster[Non_Samples[i]]]);
    }
  }
  else{
    for (i = 0; i < n_Samples; i++)	{
      value += dot_mult(p_Docs, Samples[i], Concept_Vector[Cluster[Samples[i]]]);
    }
  }
	
  return value;
}


void movmf_base::FullSelect(){
  int i;
  Samples = new int[n_Docs];
  n_Samples = n_Docs;
  avg_Docs = (double) n_Samples/n_Clusters;
  for(i=0; i< n_Samples; i++){
    Samples[i] = i;
  }
}
  
// select n_Clusters document vectors as concept vectors
void movmf_base::random_init(SparseMatrixDouble *p_Docs)
{
  int i, j;
  int *selected;

  selected = new int[n_Clusters];

  // select document vectors
  for (i = 0; i < n_Clusters; i++)
    {
      while (true)
        {
          selected[i] = rand_gen.GetUniformInt(n_Docs);
          j = 0;
          while (j < i && selected[j] != selected[i])
            j++;
          if (j == i)	// this document vector has not been selected
            break;
        }
    }

  // copy document vectors to concept vectors
  for (i = 0; i < n_Clusters; i++)
    for (j = 0; j < n_Words; j++)
      Concept_Vector[i][j] = (*p_Docs)(j, selected[i]);
}



// nor verified with classic k means ...
void movmf_base::random_perturb_init(SparseMatrixDouble *p_Docs)
{
  VECTOR_double v, temp_v;
  int i, j;
  float temp_norm;

  v = new float[n_Words];
  temp_v = new float[n_Words];

  for (i = 0; i < n_Words; i++)
    v[i] = 0.0;

  // compute the centroid of document vectors
  for (i = 0; i < n_Samples; i++)
    for (j = p_Docs->col_ptr(Samples[i]); j < p_Docs->col_ptr(Samples[i]+1); j++)
      v[p_Docs->row_ind(j)] += p_Docs->val(j);
	
  if (Alg == SPHERICAL_K_MEANS)
    normalize_vec(v, n_Words);
  else
    for (i = 0; i < n_Words; i++)
      v[i] /= n_Docs;
	
  // perturb the centroid k times with randomly generated vectors
  for (i = 0; i < n_Clusters; i++)
    {
      //generate a random vector
      for (j = 0; j < n_Words; j++)
        temp_v[j] = rand_gen.GetUniform() - 0.5;

      normalize_vec(temp_v, n_Words);
      temp_norm = Perturb * rand_gen.GetUniform();
      for (j = 0; j < n_Words; j++)
        temp_v[j] *= temp_norm;
      for (j = 0; j < n_Words; j++)
        Concept_Vector[i][j] = v[j]*(1+ temp_v[j]);
      if (Alg == SPHERICAL_K_MEANS)
        normalize_vec(Concept_Vector[i], n_Words);
	    
      // ...
      /* for (j = 0; j < n_Words; j++)
         Concept_Vector[i][j] = docs(j, i); */
    }
	
}


void movmf_base::getInitFromFile(){
  
  std::ifstream inputFile(initFile);
  
  int i,j;

  for( i = 0 ; i < n_Clusters ; i++ ){
    for( j = 0 ;  j < n_Words ; j++ ){
      inputFile >> Concept_Vector[i][j];
    }
  }
  inputFile.close();
}

/**
 * Computes internal measures of cluster quality, viz. Savg and Havg
 * Writes out results to stdout
 */
void movmf_base::ComputeInternalMeasures(SparseMatrixDouble* p)
{
  float havg = 0.0, hmin = 1.0;
  float savg = 0.0, smax = -1.0;
  float tmp;

  // Harden stuff
  assign_cluster(p);
  compute_cluster_size();

  // Make sure data is normalized
  normalize_mat(p);
  // Compute H_ave
  for (int i = 0; i < n_Samples; i++) {
    tmp = cosine(p, i, Concept_Vector[Cluster[i]], n_Words);
    havg += tmp;
    if (tmp < hmin)
      hmin = tmp;
  }
  if (!silent)
    cout << "Writing internal measures file\n";
  //cout << "Havg := " << havg/n_Samples << endl;
  //cout << "Hmin := " << hmin << endl;

  string fname("internal");
  if (isSoft) {
    fname += "SOFT";
  }
  fname += "moVMF.";
  char buf[3];
  sprintf(buf, "%d", n_Clusters);
  fname += buf;
  ofstream ofile(fname.data());
  if (ofile.fail()) {
    cerr << "WARNING: Failed to open " << fname << endl;
  }

  for (int i = 0; i < n_Clusters - 1; i++)
    for (int j = i+1; j < n_Clusters; j++) {
      tmp = cosine(Concept_Vector[i], Concept_Vector[j], n_Words);
      if (tmp > smax) smax = tmp;
      tmp *= ClusterSize[i]*ClusterSize[j];
      savg += tmp;
    }
  tmp = 0.0;
  for (int i = 0; i < n_Clusters - 1; i++)
    for (int j = i+1; j < n_Clusters; j++) {
      tmp += ClusterSize[i]*ClusterSize[j];
    }
  savg /= tmp;

  ofile << havg/n_Samples
        << " " << hmin
        << " " << savg
        << " " << smax
        << endl;
  ofile.close();
  //cout << "Savg := " << savg << endl;
  //cout << "Smax := " << smax << endl;
}
