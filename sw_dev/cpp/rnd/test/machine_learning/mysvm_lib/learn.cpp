#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <float.h>
#include "globals.h"
#include "example_set.h"
#include "svm_c.h"
#include "parameters.h"
#include "kernel.h"
#include "svm_nu.h"
#include "version.h"

#include <time.h>
#include <sys/types.h>
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
#include <unistd.h>
#endif

// global svm-objects
kernel_c* kernel=0;
parameters_c* parameters=0;
svm_c* svm;
example_set_c* training_set=0;
int is_linear=1; // linear kernel?

struct example_set_list{
  example_set_c* the_set;
  example_set_list* next;
};
example_set_list* test_sets = 0;


void print_help(){
  std::cout<<std::endl;
  std::cout<<"my_svm: train a svm from the given parameters and examples."<<std::endl<<std::endl;
  std::cout<<"usage: my_svm"<<std::endl
      <<"       my_svm <FILE>"<<std::endl
      <<"       my_svm <FILE1> <FILE2> ..."<<std::endl<<std::endl;
  std::cout<<"The input has to consist of:"<<std::endl
      <<"- the svm parameters"<<std::endl
      <<"- the kernel definition"<<std::endl
      <<"- the training set"<<std::endl
      <<"- one or more test sets (optional)"<<std::endl;

  std::cout<<std::endl<<"See the documentation for the input format. The first example set to be entered is considered to be the training set, all others are test sets. Each input file can consist of one or more definitions. If no input file is specified, the input is read from <stdin>."<<std::endl<<std::endl;

  std::cout<<std::endl<<"This software is free only for non-commercial use. It must not be modified and distributed without prior permission of the author. The author is not responsible for implications from the use of this software."<<std::endl;
  exit(0);
};


void read_input(std::istream& input_stream, char* filename){
  // returns number of examples sets read
  char* s = new char[MAXCHAR];
  char next;
  next = input_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = input_stream.get(); 
  };
  while(! input_stream.eof()){
    if('#' == next){
      // ignore comment
      input_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    (' ' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    ('\t' == next)){
      // ignore
      next = input_stream.get();
    }
    else if('@' == next){
      // new section
      input_stream >> s;
      if(0 == strcmp("@parameters",s)){
	// read parameters
	if(parameters == 0){
	  parameters = new parameters_c();
	  input_stream >> *parameters;
	}
	else{
	  std::cout <<"*** ERROR: Parameters multiply defined"<<std::endl;
	  throw input_exception();
	};
      }
      else if(0==strcmp("@examples",s)){
	if(0 == training_set){
	  // input training set
	  training_set = new example_set_c();
	  if(0 != parameters){
	    training_set->set_format(parameters->default_example_format);
	  };
	  input_stream >> *training_set;	    
	  training_set->set_filename(filename);
	  std::cout<<"   read "<<training_set->size()<<" examples, format "<<training_set->my_format<<", dimension = "<<training_set->get_dim()<<"."<<std::endl;
	}
	else{
	  // input test sets
	  example_set_list* test_set = new example_set_list;
	  test_set->the_set = new example_set_c();
	  if(0 != parameters){
	    (test_set->the_set)->set_format(parameters->default_example_format);
	  };
	  input_stream >> *(test_set->the_set);
	  (test_set->the_set)->set_filename(filename);
	  test_set->next = test_sets;
	  test_sets = test_set;
	  std::cout<<"   read "<<(test_set->the_set)->size()<<" examples, format "<<(test_set->the_set)->my_format<<", dimension = "<<(test_set->the_set)->get_dim()<<"."<<std::endl;
	};
      }
      else if(0==strcmp("@kernel",s)){
	if(0 == kernel){
	  kernel_container_c k_cont;
	  input_stream >> k_cont;
	  kernel = k_cont.get_kernel();
	  is_linear = k_cont.is_linear;
	}
	else{
	  std::cout <<"*** ERROR: Kernel multiply defined"<<std::endl;
	  throw input_exception();
	};
      };
    }
    else{
      // default = "@examples"
      if(0 == training_set){
	// input training set
	training_set = new example_set_c();
	if(0 != parameters){
	  training_set->set_format(parameters->default_example_format);
	};
	input_stream  >> *training_set;	    
	training_set->set_filename(filename);
	std::cout<<"   read "<<training_set->size()<<" examples, format "<<training_set->my_format<<", dimension = "<<training_set->get_dim()<<"."<<std::endl;
      }
      else{
	// input test sets
	example_set_list* test_set = new example_set_list;
	test_set->the_set = new example_set_c();
	if(0 != parameters){
	  (test_set->the_set)->set_format(parameters->default_example_format);
	};
	input_stream >> *(test_set->the_set);
	(test_set->the_set)->set_filename(filename);
	test_set->next = test_sets;
	test_sets = test_set;
	std::cout<<"   read "<<(test_set->the_set)->size()<<" examples, format "<<(test_set->the_set)->my_format<<", dimension = "<<(test_set->the_set)->get_dim()<<"."<<std::endl;
      };
    };
    next = input_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = input_stream.get(); 
    };
  };
  delete []s;
};


svm_result do_cv(){
  SVMINT number = parameters->cross_validation;
  SVMINT size = training_set->size();
  int verbosity = parameters->verbosity;
  if((number > size) || (0 >= number)){
    number = size; // leave-one-out testing
  };
  //  SVMINT cv_size = size / number;
  if(! parameters->cv_inorder){
    training_set->permute();
  };
  training_set->clear_alpha();
  example_set_c* cv_train=new example_set_c(); //=0;
  example_set_c* cv_test=new example_set_c(); //=0;
  svm_result train_result;
  svm_result test_result;
  svm_result train_sum;
  svm_result test_sum;
  train_sum.VCdim = 0;
  train_sum.pred_loss=0;
  train_sum.loss=0;
  train_sum.loss_pos=0;
  train_sum.loss_neg=0;
  train_sum.MAE = 0;
  train_sum.MSE = 0;
  train_sum.accuracy = 0;
  train_sum.precision = 0;
  train_sum.recall=0;
  train_sum.number_svs=0;
  train_sum.number_bsv=0;
  test_sum.VCdim = 0;
  test_sum.loss=0;
  test_sum.loss_pos=0;
  test_sum.loss_neg=0;
  test_sum.MAE = 0;
  test_sum.MSE = 0;
  test_sum.accuracy = 0;
  test_sum.precision = 0;
  test_sum.recall=0;
  test_sum.number_svs=0;
  test_sum.number_bsv=0;
  SVMINT j;
  if(verbosity>2){
    if(parameters->cv_window>0){
      std::cout<<"beginning "<<(number-parameters->cv_window)<<" sliding window steps"<<std::endl;
    }
    else{
      std::cout<<"beginning "<<number<<"-fold crossvalidation"<<std::endl;
    };
  };
  SVMINT i;
  for(i=parameters->cv_window;i<number;i++){
    // do cv
    if(verbosity >= 3){
      std::cout<<"----------------------------------------"<<std::endl;
      std::cout<<(i+1);
      if(0 == i%10) std::cout<<"st";
      else if(1==i%10) std::cout<<"nd";
      else if(2==i%10) std::cout<<"rd";
      else std::cout<<"th";
      std::cout<<" step"<<std::endl;
    };
    //std::cout<<"From "<<i*cv_size<<" to "<<(i+1)*cv_size<<std::endl;
    cv_train->clear();
    cv_test->clear();
    cv_train->set_dim(training_set->get_dim());
    cv_test->set_dim(training_set->get_dim());
    if(training_set->initialised_y()){
      cv_train->set_initialised_y();
      cv_test->set_initialised_y();
    };
    cv_train->put_Exp_Var(training_set->get_exp(),training_set->get_var());
    cv_test->put_Exp_Var(training_set->get_exp(),training_set->get_var());
    if(verbosity>=4){
      std::cout<<"Initing examples sets"<<std::endl;
    };

    if(parameters->cv_window>0){
      // training window
      for(j=SVMINT((i-parameters->cv_window)*size/number);((j<(SVMINT)(i*size/number))&&(j<size));j++){
	cv_train->put_example(training_set->get_example(j));
      };
      // test window
      for(j=(SVMINT)(i*size/number);((j<(SVMINT)((i+1)*size/number))&&(j<size));j++){
	cv_test->put_example(training_set->get_example(j));
      };
    }
    else{
      for(j=(SVMINT)(i*size/number);((j<(SVMINT)((i+1)*size/number))&&(j<size));j++){
	cv_test->put_example(training_set->get_example(j));
      };
      for(j=0;j<(SVMINT)(i*size/number);j++){
	cv_train->put_example(training_set->get_example(j));
      };
      for(j=(SVMINT)((i+1)*size/number);j<size;j++){
	cv_train->put_example(training_set->get_example(j));
      };
    };
    cv_train->clear_alpha();
    cv_test->clear_alpha();
    cv_train->compress();
    cv_test->compress();

    if(verbosity>=4){
      std::cout<<"Setting up the SVM"<<std::endl;
    };
    kernel->init(parameters->kernel_cache,cv_train);
    svm->init(kernel,parameters);

    // train & test the svm
    if(verbosity>=4){
      std::cout<<"Training"<<std::endl;
    };
    //    cv_train->clear_alpha();

    train_result = svm->train(cv_train);
    if(verbosity>=4){
      std::cout<<"Testing"<<std::endl;
    };
    test_result = svm->test(cv_test,0);

    train_sum.VCdim += train_result.VCdim;
    train_sum.loss += train_result.loss;
    train_sum.loss_pos += train_result.loss_pos;
    train_sum.loss_neg += train_result.loss_neg;
    train_sum.MAE += train_result.MAE;
    train_sum.MSE += train_result.MSE;
    train_sum.pred_loss += train_result.pred_loss;
    train_sum.accuracy += train_result.accuracy;
    train_sum.precision += train_result.precision;
    train_sum.recall += train_result.recall;
    train_sum.number_svs += train_result.number_svs;
    train_sum.number_bsv += train_result.number_bsv;
    test_sum.loss += test_result.loss;
    test_sum.loss_pos += test_result.loss_pos;
    test_sum.loss_neg += test_result.loss_neg;
    test_sum.MAE += test_result.MAE;
    test_sum.MSE += test_result.MSE;
    test_sum.accuracy += test_result.accuracy;
    test_sum.precision += test_result.precision;
    test_sum.recall += test_result.recall;
    if(verbosity>=4){
      std::cout<<"Training set:"<<std::endl
	  <<"Loss: "<<train_result.loss<<std::endl;
      if(parameters->Lpos != parameters->Lneg){
	std::cout<<"  Loss+: "<<train_result.loss_pos<<std::endl;
	std::cout<<"  Loss-: "<<train_result.loss_neg<<std::endl;
      };
      std::cout<<"MAE: "<<train_result.MAE<<std::endl;
      std::cout<<"MSE: "<<train_result.MSE<<std::endl;
      std::cout<<"VCdim: "<<train_result.VCdim<<std::endl;
      if(parameters->is_pattern){
	std::cout<<"Accuracy  : "<<train_result.accuracy<<std::endl
	    <<"Precision : "<<train_result.precision<<std::endl
	    <<"Recall    : "<<train_result.recall<<std::endl;
      };
      std::cout<<"Support Vectors : "<<train_result.number_svs<<std::endl;
      std::cout<<"Bounded SVs     : "<<train_result.number_bsv<<std::endl;
      std::cout<<"Test set:"<<std::endl
	  <<"Loss: "<<test_result.loss<<std::endl;
      if(parameters->Lpos != parameters->Lneg){
	std::cout<<"  Loss+: "<<test_result.loss_pos<<std::endl;
	std::cout<<"  Loss-: "<<test_result.loss_neg<<std::endl;
      };
      if(parameters->is_pattern){
	std::cout<<"Accuracy  : "<<test_result.accuracy<<std::endl
	    <<"Precision : "<<test_result.precision<<std::endl
	    <<"Recall    : "<<test_result.recall<<std::endl;
      };
    };
  };
  parameters->verbosity = verbosity;
  number-=parameters->cv_window;
  if(verbosity > 1){
    std::cout<<"----------------------------------------"<<std::endl;
    std::cout<<"Results of "<<number<<"-fold cross-validation:"<<std::endl;
    std::cout<<"-- Training set: --"<<std::endl
	<<"Loss: "<<train_sum.loss/number<<std::endl;
    if(parameters->Lpos != parameters->Lneg){
      std::cout<<"  Loss+: "<<train_sum.loss_pos/number<<std::endl;
      std::cout<<"  Loss-: "<<train_sum.loss_neg/number<<std::endl;
    };
    std::cout<<"MAE: "<<train_sum.MAE/number<<std::endl;
    std::cout<<"MSE: "<<train_sum.MSE/number<<std::endl;
    std::cout<<"VCdim: "<<train_sum.VCdim/number<<std::endl;
    if(parameters->is_pattern){
      std::cout<<"Accuracy  : "<<train_sum.accuracy/number<<std::endl
	  <<"Precision : "<<train_sum.precision/number<<std::endl
	  <<"Recall    : "<<train_sum.recall/number<<std::endl;
    };
    std::cout<<"Support Vectors : "<<((SVMFLOAT)train_sum.number_svs)/number<<std::endl;
    std::cout<<"Bounded SVs     : "<<((SVMFLOAT)train_sum.number_bsv)/number<<std::endl;
    std::cout<<"-- Test set: --"<<std::endl
	<<"Loss: "<<test_sum.loss/number<<std::endl;
    if(parameters->Lpos != parameters->Lneg){
      std::cout<<"  Loss+: "<<test_sum.loss_pos/number<<std::endl;
      std::cout<<"  Loss-: "<<test_sum.loss_neg/number<<std::endl;
    };
    std::cout<<"MAE: "<<test_sum.MAE/number<<std::endl;
    std::cout<<"MSE: "<<test_sum.MSE/number<<std::endl;

    if(parameters->is_pattern){
      std::cout<<"Accuracy  : "<<test_sum.accuracy/number<<std::endl
	  <<"Precision : "<<test_sum.precision/number<<std::endl
	  <<"Recall    : "<<test_sum.recall/number<<std::endl;
    };
  };
  test_sum.VCdim = train_sum.VCdim/number;
  test_sum.loss /= number;
  test_sum.loss_pos /= number;
  test_sum.loss_neg /= number;
  test_sum.MAE /= number;
  test_sum.MSE /= number;
  test_sum.pred_loss = train_sum.pred_loss;
  test_sum.accuracy /= number;
  test_sum.precision /= number;
  test_sum.recall /= number;
  test_sum.number_svs = train_sum.number_svs/number;
  test_sum.number_bsv = train_sum.number_bsv/number;
  delete cv_test;
  delete cv_train;

  return test_sum;
};


svm_result train(){
  svm_result the_result;
  if(parameters->cross_validation > 0){
    the_result = do_cv();
  }
  else{
    kernel->init(parameters->kernel_cache,training_set);
    svm->init(kernel,parameters);

    if(parameters->is_nu || parameters->is_distribution){
      std::cout<<"Training started with nu = "
	  <<parameters->nu
	  <<"."<<std::endl;
    }
    else if(parameters->get_Cpos() == parameters->get_Cneg()){
      std::cout<<"Training started with C = "
	  <<parameters->get_Cpos()
	  <<"."<<std::endl;
    }
    else{
      std::cout<<"Training started with C = ("<<parameters->get_Cpos()
	  <<","<<parameters->get_Cneg()<<")."<<std::endl;
    };
    the_result = svm->train(training_set);
  };
  return the_result;
};


inline
SVMFLOAT to_minimize(svm_result result){
  // which value to minimize in calc_c
  if((parameters->cross_validation <= 0) && (1 == parameters->is_pattern)){
    return (1.0-result.pred_accuracy);
  }
  else{
    if(1 == parameters->is_pattern){
      return (1.0-result.accuracy);
    }
    else{
      return result.loss;
    };
  };
};

svm_result calc_c(){
  const SVMFLOAT lambda = 0.618033989; // (sqrt(5)-1)/2
  SVMINT verbosity = parameters->verbosity;
  parameters->verbosity -= 2;
  svm_result the_result;
  SVMFLOAT c_min = parameters->c_min;
  SVMFLOAT c_max = parameters->c_max;
  SVMFLOAT c_delta = parameters->c_delta;
  SVMFLOAT oldC;
  SVMINT last_dec=0; // when did loss decrease?
  // setup s,t
  if(verbosity >= 3){
    std::cout<<"starting search for C"<<std::endl;
  };
  if((parameters->search_c == 'a') ||(parameters->search_c == 'm')){
    SVMFLOAT minimal_value=infinity;
    SVMFLOAT minimal_C=c_min;
    svm_result minimal_result;
    SVMFLOAT result_value;
    oldC=c_min;
    training_set->clear_alpha();
    while(c_min <= c_max){
      if(verbosity>=3){
	std::cout<<"C = "<<c_min<<" :"<<std::endl;
      };
      parameters->realC = c_min;
      training_set->scale_alphas(c_min/oldC);
      // training_set->clear_alpha();
      oldC = c_min;
      if(verbosity >= 4){
	std::cout<<"C = "<<c_min<<std::endl;
      }
      the_result = train();
      if(verbosity>=3){
	std::cout<<"loss = "<<the_result.loss<<std::endl;
	if(parameters->is_pattern){
	  std::cout<<"accuracy = "<<the_result.accuracy<<std::endl;
	  // std::cout<<"predicted loss = "<<the_result.pred_loss<<std::endl;
	  std::cout<<"predicted accuracy = "<<the_result.pred_accuracy<<std::endl;
	};
	//	std::cout<<"VCdim <= "<<the_result.VCdim<<std::endl;
      };
      result_value = to_minimize(the_result);
      //      std::cout<<result_value<<std::endl;
      last_dec++;
#if defined(WIN32) || defined(_WIN32)
      if((result_value<minimal_value) && (! _isnan(result_value))){
#else
      if((result_value<minimal_value) && (! isnan(result_value))){
#endif
	minimal_value=result_value;
	minimal_C=c_min;
	minimal_result = the_result;
	last_dec=0;
      };
      if(parameters->search_c == 'a'){
	c_min += c_delta;
      }
      else{
	c_min *= c_delta;
      };
      if((parameters->search_stop > 0) && (last_dec >= parameters->search_stop)){
	// no decrease in loss, stop
	c_min = 2*c_max;
      };
    };
    parameters->realC = minimal_C;
    the_result = minimal_result;
  }
  else{
    // method of golden ratio
    
    SVMFLOAT s = lambda*c_min+(1-lambda)*c_max;
    SVMFLOAT t = (1-lambda)*c_min+lambda*c_max;
    SVMFLOAT phi_s;
    SVMFLOAT phi_t;
    
    parameters->realC = s;
    training_set->clear_alpha();
    the_result = train();
    phi_s = to_minimize(the_result);
    parameters->realC = t;
    training_set->scale_alphas(t/s);
    oldC = t;
    the_result = train();
    phi_t = to_minimize(the_result);
    while(c_max - c_min > c_delta*c_min){
      if(verbosity >= 3){
	std::cout<<"C in ["<<c_min<<","<<c_max<<"]"<<std::endl;
      };
      if(phi_s < phi_t){
	c_max = t;
	t = s;
	phi_t = phi_s;
	// calc s
	s = lambda*c_min+(1-lambda)*c_max;
	parameters->realC = s;
	training_set->scale_alphas(s/oldC);
	oldC=s;
	the_result = train();
	phi_s = to_minimize(the_result);
      }
      else{
	c_min = s;
	s = t;
	phi_s = phi_t;
	// calc t
	t = (1-lambda)*c_min+lambda*c_max;
	parameters->realC = t;
	training_set->scale_alphas(t/oldC);
	oldC=t;
	the_result = train();
	phi_t = to_minimize(the_result);
      };
    };
    // save last results
    if(phi_s < phi_t){
      c_max = t;
    }
    else{
      c_min = s;
    };
    parameters->realC = (c_min+c_max)/2;

  };

  // ouput result
  if(verbosity >= 1){
    std::cout<<"*** Optimal C is "<<parameters->realC;
    if(parameters->search_c == 'g'){
      std::cout<<" +/-"<<((c_max-c_min)/2);
    };
    std::cout<<std::endl;
  };
  if(verbosity>=2){
    std::cout<<"result:"<<std::endl
	<<"Loss: "<<the_result.loss<<std::endl;
    if(parameters->Lpos != parameters->Lneg){
      std::cout<<"  Loss+: "<<the_result.loss_pos<<std::endl;
      std::cout<<"  Loss-: "<<the_result.loss_neg<<std::endl;
    };
    if(parameters->is_pattern){
      std::cout<<"predicted Loss: "<<the_result.pred_loss<<std::endl;
    };
    std::cout<<"MAE: "<<the_result.MAE<<std::endl;
    std::cout<<"MSE: "<<the_result.MSE<<std::endl;
    std::cout<<"VCdim <= "<<the_result.VCdim<<std::endl;

    if(parameters->is_pattern){
      std::cout<<"Accuracy  : "<<the_result.accuracy<<std::endl
	  <<"Precision : "<<the_result.precision<<std::endl
	  <<"Recall    : "<<the_result.recall<<std::endl;
      if(parameters->cross_validation == 0){
	std::cout<<"predicted Accuracy  : "<<the_result.pred_accuracy<<std::endl
	    <<"predicted Precision : "<<the_result.pred_precision<<std::endl
	    <<"predicted Recall    : "<<the_result.pred_recall<<std::endl;
      };
    };
    std::cout<<"Support Vectors : "<<the_result.number_svs<<std::endl;
    std::cout<<"Bounded SVs     : "<<the_result.number_bsv<<std::endl;
    if(parameters->search_c == 'g'){
      std::cout<<"(WARNING: this is the last result attained and may slightly differ from the result of the optimal C!)"<<std::endl;
    };
  };
  parameters->verbosity = verbosity;
  return the_result;
};


///////////////////////////////////////////////////////////////



int main(int argc,char* argv[]){
  std::cout<<"*** mySVM version "<<mysvmversion<<" ***"<<std::endl;
  std::cout.precision(8);
  // read objects
  try{
    if(argc<2){
      std::cout<<"Reading from STDIN"<<std::endl;
      // read vom std::cin
      read_input(std::cin,"mysvm");
    }
    else{
      char* s = argv[1];
      if((0 == strcmp("-h",s)) || (0==strcmp("-help",s)) || (0==strcmp("--help",s))){
	// print out command-line help
	print_help();
      }
      else{
	// read in all input files
	for(int i=1;i<argc;i++){
	  if(0 == strcmp(argv[i],"-")){
	    std::cout<<"Reading from STDIN"<<std::endl;
	    // read vom std::cin
	    read_input(std::cin,"mysvm");
	  }
	  else{
	    std::cout<<"Reading "<<argv[i]<<std::endl;
	    std::ifstream input_file(argv[i]);
	    if(input_file.bad()){
	      std::cout<<"ERROR: Could not read file \""<<argv[i]<<"\", exiting."<<std::endl;
	      exit(1);
	    };
	    read_input(input_file,argv[i]);
	    input_file.close();
	  };
	};
      };
    };
  }
  catch(general_exception &the_ex){
    std::cout<<"*** Error while reading input: "<<the_ex.error_msg<<std::endl;
    exit(1);
  }
  catch(...){
    std::cout<<"*** Program ended because of unknown error while reading input"<<std::endl;
    exit(1);
  };

  if(0 == parameters){
    parameters = new parameters_c();
    if(training_set->initialised_pattern_y()){
      parameters->is_pattern = 1;
      parameters->do_scale_y = 0;
    };
  };
  parameters->is_linear = is_linear;
  if(0 == kernel){
    kernel = new kernel_dot_c();
  };
  if(0 == training_set){
    std::cout << "*** ERROR: You did not enter the training set"<<std::endl;
    exit(1);
  };
  if(2 > training_set->size()){
    std::cout << "*** ERROR: Need at least two examples to learn."<<std::endl;
    exit(1);
  };

  if(parameters->is_distribution){
    svm = new svm_distribution_c();
    std::cout<<"distribution estimation SVM generated"<<std::endl;
  }
  else if(parameters->is_nu){
    if(parameters->is_pattern){
      svm = new svm_nu_pattern_c();
      if(! training_set->initialised_pattern_y()){
	std::cout<<"WARNING: Parameters set a pattern SVM, but the training ys are not in {-1,1}."<<std::endl;
      }
      std::cout<<"nu-PSVM generated"<<std::endl;
    }
    else{
      svm = new svm_nu_regression_c();
      std::cout<<"nu-RSVM generated"<<std::endl;
    };
  }
  else if(parameters->is_pattern){
    svm = new svm_pattern_c();
    if(! training_set->initialised_pattern_y()){
      std::cout<<"WARNING: Parameters set a pattern SVM, but the training ys are not in {-1,1}."<<std::endl;
    }
    std::cout<<"PSVM generated"<<std::endl;
  }
  else{
    svm = new svm_regression_c();
    std::cout<<"RSVM generated"<<std::endl;
  };

  // scale examples
  if(parameters->do_scale){
    training_set->scale(parameters->do_scale_y);
  };

  // training the svm
  if(parameters->search_c != 'n'){
    calc_c();
    std::cout<<"re-training without CV and C = "<<parameters->realC<<std::endl; 
    parameters->cross_validation = 0;
    parameters->verbosity -= 1;
    train();
    parameters->verbosity += 1;
  }
  else{
    train();
  };


  if(0 == parameters->cross_validation){
    // save results
    if(parameters->verbosity > 1){
      std::cout<<"Saving trained SVM to "<<(training_set->get_filename())<<".svm"<<std::endl;
    };
    char* outname = new char[MAXCHAR];
    strcpy(outname,training_set->get_filename());
    strcat(outname,".svm");
    std::ofstream output_file(outname,std::ios::out|std::ios::trunc);
    output_file.precision(16);
    output_file<<*training_set;
    output_file.close();
    delete []outname;
  };

  // testing
  if((parameters->cross_validation > 0) && (0 != test_sets)){
    // test result of cross validation: train new SVM on whole example set
    parameters->cross_validation = 0;
    std::cout<<"Re-training SVM on whole example set for testing"<<std::endl;
    train();
  };
  if(0 != test_sets){
    std::cout<<"----------------------------------------"<<std::endl;
    std::cout<<"Starting tests"<<std::endl;
    example_set_c* next_test;
    SVMINT test_no = 0;
    char* outname = new char[MAXCHAR];
    while(test_sets != 0){
      test_no++;
      next_test = test_sets->the_set;
      if(parameters->do_scale){
	next_test->scale(training_set->get_exp(),
			 training_set->get_var(),
			 training_set->get_dim());
      };
      if(next_test->initialised_y()){
	std::cout<<"Testing examples from file "<<(next_test->get_filename())<<std::endl;
	svm->test(next_test,1);
      }
      else{
	std::cout<<"Predicting examples from file "<<(next_test->get_filename())<<std::endl;
	svm->predict(next_test);
	// output to file .pred

	strcpy(outname,next_test->get_filename());
	strcat(outname,".pred");
	std::ofstream output_file(outname,std::ios::out);
	output_file<<"@examples"<<std::endl;
	output_file<<(*next_test);
	output_file.close();	
      };
      test_sets = test_sets->next; // skip delete!
    };
    delete []outname;
  };

  if(kernel) delete kernel;
  delete svm;
  if(parameters->verbosity > 1){
    std::cout << "mysvm ended successfully."<<std::endl;
  };
  return(0);
};
