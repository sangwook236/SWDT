// movmf.h -- driver class for doing movmf based clustering
// Author: Suvrit Sra

// Copyright (C) 2005,2006,2007 Suvrit Sra (suvrit@cs.utexas.edu)
// Copyright The University of Texas at Austin

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.


#ifndef _MOVMF_H_
#define _MOVMF_H_

#include "matrix.h"
#include "timerUtil.h"
#include "SparseMatrixDouble.h"
#include "movmf_base.h"

class movmf {
  int mymc;
  int argc;
  char** argv;
  int *cluster;
  doc_mat mat;
  bool no_args;
  clock_t start_clock, finish_clock;

  int alg, K_Clusters, n_Clusters, kappa;

  int init_method;
  float perturb, epsilon, coherence;

  int num_sample, emptyCluster;
  float balance;

  int encode_scheme, objfun, kmax;

  // output control
  bool no_output, dump, full_dump, decomposition, soft_assign;

  int  scheme, meth;

  char ver[MAX_VER_DIGIT_LENGTH+1];
  char output_matrix[256];
  char wordCluster[256];
  char suffix[128];
  char docs_matrix[256];
  char initFile[256];
  char version[256];
  char postfix[20];
	
  SparseMatrixDouble* data;
  movmf_base* movmf_obj;
  TimerUtil runtime_est;

public:
  movmf(int argc, char** argv);
  int run();

private:
  int process_command_line();
  int continue_running();
  int conclude();
  int dump_to_file();
  int dump_to_stdout();
  void read_mat(char*, char*, doc_mat*);
  void print_help_basic();
  void print_help_adv  ();
};

#endif // _MOVMF_H_
