// File: movmf.cc
// Author: Suvrit Sra
// Time-stamp: <06 November 2007 11:16:13 AM CET --  suvrit>

/* Derived from the original SPKMeans.h code written by Yuqiang Guan
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

#include <iostream>
#include <time.h>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

#include "mat_vec.h"
#include "movmf_base.h"
#include "timerUtil.h"
#include "matrix.h"
#include "movmf.h"

movmf::movmf(int ac, char **av)
{
  argc = ac; argv = av;
  cluster = 0;
  no_args = true;
  coherence = 0;

  // parameters for the k-means algorithm
  alg = SPHERICAL_K_MEANS;
  K_Clusters=0; n_Clusters = 3; kappa = 1;

  epsilon = DEFAULT_EPSILON; 

  init_method = RANDOM_PERTURB_INIT; 
  perturb = DEFAULT_PERTURB;
  num_sample = 0; balance = 1.0;
  encode_scheme = NORM_TERM_FREQ; objfun = NON_WEIGHTED_OBJ;
  kmax = 1000;

  // output control
  no_output = false; dump = false; full_dump = false; 
  decomposition=false; soft_assign = false;

  scheme = SPKMEANS; meth = CON_DEC ;

  ver[0] = '\0';
  strcpy(output_matrix, "output_matrix");
  strcpy(wordCluster,"wordClusterFile");
  strcpy(suffix, "tfn");
  strcpy(initFile, "data_Init");
  strcpy(version, "2.0");
}

int movmf::run()
{
  int result;
  if ( (result = process_command_line()) < 0)
    return result;
  if ( (result = continue_running()) < 0)
    return result;
  
  conclude();
  if (dump)
    dump_to_file();
  if (!no_output)
    dump_to_stdout();
  return result;
}

//////////////////////////////////////////////////////////////
//----------------- Read commandline arguments ---------------
int movmf::process_command_line()
{
  //start_clock = clock();
  for (argv++; *argv != NULL; argv++) {
    if ((*argv)[0] == '-') {
      switch ((*argv)[1]) {
      case 'a':
        if ((*(++argv))[0] == 's') {
          alg = SPHERICAL_K_MEANS;
          soft_assign = true;
        } else if ((*(argv))[0] == 'h') {
          alg = SPHERICAL_K_MEANS;
          soft_assign = false;
        } else {
          cerr << "Invalid algorithm" << *argv << endl;
          print_help_adv();
          return -1;
        }
        break;
      case 'i':
        switch ((*(++argv))[0]) {
        case 's':
          init_method = SAMPLE_INIT;
          break;
        case 'p':
          init_method = RANDOM_PERTURB_INIT;
          break;
        case 'r':
          init_method = RANDOM_INIT;
          break;
        case 'f':
          init_method = FROM_FILE;
          strcpy(initFile, *(++argv));
          break;
        default:
          cerr << "Invalid option: " << *argv << endl;
          print_help_adv();
          return -1;
        }
        break;
      case 'c':
        n_Clusters = atoi(*(++argv));
        break;
      case 'K':
        kappa = atoi(*(++argv));
        break;
      case 'Z':
        kmax = atoi(*(++argv));
        break;
      case 't':
        strcpy(suffix, *(++argv));
        break;
      case 'e':
        epsilon = atof(*(++argv));
        break;
      case 's':
        no_output = true;
        break;
      case 'v':
        //strncpy(ver, *(++argv), MAX_VER_DIGIT_LENGTH);
        cout << "moVMF Version: " << version << endl 
             << "(C) 2005--CURRENT. Arindam Banerjee & Suvrit Sra.\n";

        return -1;
      case 'n':
        dump = false;
        break;
      case 'h':
        print_help_adv();
        return -1;
      case 'd':
        full_dump = true;
        break;
      case 'O':
        strcpy(output_matrix, *(++argv));
        strcpy(wordCluster, output_matrix);
        break;
      case 'p':
        perturb = atof(*(++argv));
        break;
      case 'N':
        num_sample = atoi(*(++argv));
        break;
      case 'B':
        balance = atof(*(++argv));
        break;
      case 'E':
        encode_scheme = atoi(*(++argv));
        break;
      default:
        printf("Invalid switch %s\n", *argv);
        print_help_adv();
        return -1;
      }
    }
    else
      {
        sprintf(docs_matrix, "%s", *argv);
        read_mat(*argv,suffix,  &mat);
        no_args = false;
      }
  }
	
  if (no_args)  {
    print_help_basic();
    return -1;
  }
  return 0;
}

int movmf::continue_running()
{
  //////////////////////////////////////////////////////////////
	
  data = new SparseMatrixDouble(mat.n_row, mat.n_col, mat.n_nz, mat.val, mat.row_ind, mat.col_ptr); 
	
  // clustering
  start_clock = clock();
  if (alg == SPHERICAL_K_MEANS)
    normalize_mat(data);
  if (K_Clusters == 0)
    K_Clusters=n_Clusters;
	
  // will have to pass soft_assign and appropriate type of cluster
  movmf_obj = new movmf_base(data, soft_assign, K_Clusters, kappa, init_method, initFile, epsilon,kmax);

  movmf_obj->set_memory_consume(mymc);
  movmf_obj->set_silent(no_output);

  if(!soft_assign){
    cluster = new int[mat.n_col];
    movmf_obj->Init(cluster);
  }

  movmf_obj->SetAlgorithm(alg);
  movmf_obj->SetDump(!no_output & full_dump);
  if (init_method == RANDOM_PERTURB_INIT)
    movmf_obj->SetPerturb(perturb);
  else if (init_method == SAMPLE_INIT)
    movmf_obj->SetNumSamples(num_sample);
  movmf_obj->SetEncodeScheme(encode_scheme);
  movmf_obj->SetObjFunc(objfun);

  if (!no_output)
    cout << " initial part over \n";
  movmf_obj->FullSelect();

  if (!no_output)
    cout << "*********** k = "<< n_Clusters << " *************\n";
  movmf_obj->InitIterate(data);

  if (!no_output)
    cout << "------- Init done --------" << endl;

  movmf_obj->Iterate(data,soft_assign);

  if (!no_output)
    cout << "--- done ---\n";
  finish_clock = clock();

  return 0;
}

int movmf::conclude()
{
  //   --------------- do not change anything below this (for now) ---------------
	

  if (n_Clusters <=0)
    n_Clusters = K_Clusters;

  sprintf(postfix, ".%d", n_Clusters);
  if (strcmp(ver, "") != 0)
	{
      strcat(postfix, ".");
      strcat(postfix, ver);
	}

  //int *cluster_size = new int[n_Clusters]; 
	
  emptyCluster=0;
  if(!soft_assign){
    int *cluster_size = new int[n_Clusters]; 
    emptyCluster = movmf_obj->SortCluster(cluster_size);
    coherence = movmf_obj->GetResult();
  }
	
  if ( (scheme != QRDSPK) && (!soft_assign) ) {
    if (!no_output)
      cout << "\nWriting word cluster file" << endl;

    strcat(wordCluster, "_");
    strcat(wordCluster, suffix);
    movmf_obj->wordCluster(wordCluster, n_Clusters-emptyCluster);
  }
  char browserfilepost[256];
	
  sprintf(browserfilepost, "_doctoclus.%d", n_Clusters-emptyCluster);
  strcat(output_matrix, "_");
  strcat(output_matrix, suffix);
  strcat(output_matrix, browserfilepost);
  strcat(docs_matrix, "_docs");

  std::ofstream o_m(output_matrix);
  std::ifstream docs(docs_matrix);

  //for (i=0; i < n_Clusters; i++) cluster_size[i] = 0;
  char ch;
  int i, j;
  o_m<<mat.n_col<<endl;
  for (i = 0; i < mat.n_col; i++){
    docs>>j;
    docs>>ch;
    //docs>>ch;
    if (soft_assign){
      for (j = 0; j < n_Clusters; j++ ){
        o_m << movmf_obj->GetMembership(i,j) << " ";
      }
    }
    else{
      o_m << cluster[i];
      o_m << "\tC"<<cluster[i]<<"\t";
    }
    docs >> docs_matrix;
    o_m << docs_matrix << endl;
	  
  }
  o_m.close();
  movmf_obj->ComputeInternalMeasures(data);
  //finish_clock = clock();
  return 0;
}

int movmf::dump_to_file()
{
  char fname[MAX_DESC_STR_LENGTH+20];
   std::ofstream dump_file;
   int i;
   sprintf(fname, "%scluster%s", mat.strDesc, postfix);
   dump_file.open(fname, ios::app);
   for (i = 0; i < mat.n_col-1; i++)
  dump_file << cluster[i] << "\t";
   dump_file << cluster[mat.n_col-1] << "\n"; 
   dump_file.close();
		
   sprintf(fname, "%sobj%s", mat.strDesc, postfix);
   dump_file.open(fname, ios::app);
   dump_file << coherence << "\n"; 
   dump_file.close();
		
   sprintf(fname, "%siter%s", mat.strDesc, postfix);
   dump_file.open(fname, ios::app);
   dump_file << movmf_obj->GetNumIterations() << "\n"; 
   dump_file.close();
		
   sprintf(fname, "%scsize%s", mat.strDesc, postfix);
   dump_file.open(fname, ios::app);
   for (i = 0; i < n_Clusters-1; i++)
  dump_file << movmf_obj->GetClusterSize(i) << "\t";
   dump_file << movmf_obj->GetClusterSize(n_Clusters-1) << "\n"; 
   dump_file.close();
		
   sprintf(fname, "%stime%s", mat.strDesc, postfix);
   dump_file.open(fname, ios::app);
   dump_file << (finish_clock-start_clock)/(1e6*movmf_obj->GetNumIterations()) << "\n"; 
   dump_file.close();

   return 0;
}


int movmf::dump_to_stdout()
{

  if (alg == SPHERICAL_K_MEANS)
    cout << "spherical k means algorithm\n";
  else
    cout << "euclidean k means algorithm\n";
  if( scheme == QRDSPK)
    cout<< "First dimension reduction then clustering"<<endl;
		
  cout << "Final number of clusters: " << n_Clusters << "\n";	
  cout << "Number of documents: " << mat.n_col << "\n";
  if( scheme == QRDSPK)
    cout << "Final number of words: " << K_Clusters << "\n";
  else
    cout << "Number of words: " << mat.n_row<<endl;

  cout << "epsilon: " << epsilon << "\n";
  cout << "initialization method: ";

  switch (init_method)
    {
    case RANDOM_PERTURB_INIT:
      cout << "random perturbation\n";
      cout << "perturbation magnitude: " << perturb << "\n";
      break;
    case SAMPLE_INIT:
      cout << "subsampling\n";
      cout << "number of samples: " << num_sample << "\n";
      break;
    case RANDOM_INIT:
      cout << "totally random\n";
      break;
    case FROM_FILE:
      cout<<"cluster by reading seeds from file "<< initFile <<endl;
      break;
    default:
      break;
    }

  cout << "encoding scheme: ";
  if (encode_scheme == NORM_TERM_FREQ)
    cout << "normalized term frequency\n";
  else
    cout << "normalized term frequency inverse document\
				frequency\n";
  cout << "objective function: ";
  if (objfun == 1)
    cout << "nonweighted\n";
  else
    cout << "weighted\n";

  for (int i = emptyCluster; i < n_Clusters; i++)
    cout << "cluster " << i-emptyCluster << " : " << movmf_obj->GetClusterSize(i) << "\n";

  cout << "number of iterations: " << movmf_obj->GetNumIterations() << "\n";
  cout << "objective function value: " << coherence << "\n"; 

  runtime_est.setStopTime(cout, "computation time: ", movmf_obj->GetNumIterations());
  cout << "Output matrix file is: " << output_matrix << endl;

  cout << "Memory consumed :" << movmf_obj->get_memory_consume() << endl;
  return 0;
}

void movmf::print_help_basic()
{
  printf("USAGE: moVMF [options] input_word_doc_file\n");
}

void movmf::print_help_adv()
{
  printf("moVMF: (c) Arindam Banerjee and Suvrit Sra. 2005--NOW\n\n");
  printf("USAGE: moVMF [switches] word-doc-file\n");
  printf("\t-a algorithm\n");
  printf("\t   s: soft moVMF algorithm\n");
  printf("\t   h: hard moVMF algorithm (default)\n");
  printf("\t-i [s|p|r]\n");
  printf("\t   initialization method:\n");
  printf("\t      s -- subsampling\n");
  printf("\t      p -- random perturbation (default)\n");
  printf("\t      r -- totally random\n");
  printf("\t      f -- read from file\n");
  printf("\t-c number-of-clusters (default = 3)\n");
  printf("\t-e epsilon (default = 1e-3)\n");
  printf("\t-s   suppress output (default = false)\n");
  printf("\t-h   show this help message\n");
  printf("\t-v version-number\n");
  printf("\t-n  no dump (default)\n");
  printf("\t-d  dump the clustering process\n");
  printf("\t-p perturbation-magnitude (default = 0.1)\n");
  printf("\t   the distance between initial concept vectors\n");
  printf("\t     and the centroid will be less than this.\n");
  printf("\t-N number-of-samples (default = 0)\n");
  printf("\t-O prefix for the output clustering files\n");
  printf("\t-t scaling scheme (default = tfn)\n");
  printf("\t-K lower bound on Kappa (default = 1)\n");
  printf("\t-Z upper bound on Kappa (default = 1000)\n");
  printf("\t-E encoding-scheme\n");
  printf("\t   1: normalized term frequency (default)\n");
  printf("\t   2: normalized term frequency inverse document frequency\n");
  printf("\nExample:\n");
  printf("./moVMF -c 4 -O clusters -a s -K 20 -Z 4000 large_data\n");
}

void movmf::read_mat(char *fname, char *scaling, doc_mat *mat)
{
  char filename[256];
  clock_t start_clock, finish_clock;

  sprintf(filename, "%s%s", fname, "_dim");
  std::ifstream dimfile(filename);
  if(dimfile==0)
    cout<<"Dim file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_row_ccs");
  std::ifstream rowfile(filename);
  if(rowfile==0)
    cout<<"Row file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_col_ccs");
  std::ifstream colfile(filename);
  if(colfile==0)
    cout<<"Column file "<<filename<<" can't open.\n"<<endl;
  sprintf(filename, "%s%s", fname, "_");
  sprintf(filename, "%s%s", filename, scaling);
  sprintf(filename, "%s%s", filename, "_nz");
  std::ifstream nzfile(filename);
  if(nzfile==0)
    cout<<"Entry file "<<filename<<" can't open.\n"<<endl;

  int i;

  if(dimfile==0 || colfile==0||rowfile==0||nzfile==0)
	{
	  //cout<<"Matrix file "<< fname<<" can't open."<<endl;
	  exit(1);
	}
  //data.width(MAX_DESC_STR_LENGTH);
  //data >> mat->strDesc;
  //data >> mat->n_col >> mat->n_row >> mat->n_nz;
  start_clock = clock();
  //	cout<<"Reading the _dim file..."<<endl;
  dimfile >>mat->n_row>> mat->n_col>>mat->n_nz;
  dimfile.close();

  mat->col_ptr = new int[mat->n_col+1];
  mat->row_ind = new int[mat->n_nz];
  mat->val = new float[mat->n_nz];

  mymc =(mat->n_col+1+mat->n_nz)*sizeof(int)+mat->n_nz*sizeof(float);


  //	cout<<"Reading the _col file..."<<endl;
  for (i = 0; i < mat->n_col+1; i++)
    colfile >> (mat->col_ptr)[i];
  colfile.close();

  //	cout<<"Reading the _row file..."<<endl;
  for (i = 0; i < mat->n_nz; i++)
    rowfile >> (mat->row_ind)[i];
  rowfile.close();

  //	cout<<"Reading the _nz file..."<<endl;
  for (i = 0; i < mat->n_nz; i++)
    nzfile >> (mat->val)[i];
  nzfile.close();
  finish_clock = clock();
  //	cout<<"Reading file time: "<<(finish_clock - start_clock)/1e6<<" seconds."<<endl;
  //	cout<<"Now clustering..."<<endl;
	
  //data.close();
  // validity check ...
  // use exception handling?...
}
