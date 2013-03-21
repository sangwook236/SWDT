#include "svm_c.h"

/**
 *
 * svm_c
 *
 */

svm_c::svm_c(){
  // initialise arrays
  sum =0;
  primal=0;
  which_alpha = 0;
  lambda_eq=0;
  sum_alpha = 0;
  at_bound=0;
  all_alphas=0;
  all_ys=0;
  working_set=0;
  working_set_values=0;
  time_init=0;
  time_optimize=0;
  time_convergence=0;
  time_update=0;
  time_calc=0;
  qp.m = 1;
};


void svm_c::init(kernel_c* new_kernel, parameters_c* new_parameters){
  sum =0;
  primal=0;
  which_alpha = 0;
  lambda_eq=0;
  at_bound=0;
  target_count=0;
  kernel = new_kernel;
  parameters = new_parameters;
  Cpos = parameters->get_Cpos();
  Cneg = parameters->get_Cneg();
  is_zero = parameters->is_zero; 
  is_pattern = parameters->is_pattern;
  epsilon_pos = parameters->epsilon_pos;
  epsilon_neg = parameters->epsilon_neg;
  working_set_size = parameters->working_set_size;
  sum_alpha = 0;
  convergence_epsilon = parameters->convergence_epsilon;
  feasible_epsilon = convergence_epsilon;
  shrink_const = parameters->shrink_const;
  time_init=0;
  time_optimize=0;
  time_convergence=0;
  time_update=0;
  time_calc=0;
  qp.m = 1;
  biased = parameters->biased;
};

void svm_c::init_optimizer(){
  if(0 != sum) delete []sum;
  if(0 != which_alpha) delete []which_alpha;
  if(0 != at_bound) delete []at_bound;
  sum = new SVMFLOAT[examples_total];
  at_bound = new SVMINT[examples_total];

  // init variables
  if(working_set_size>examples_total) working_set_size = examples_total;

  qp.n = working_set_size;
  qp.c = new SVMFLOAT[qp.n];
  qp.H = new SVMFLOAT[qp.n*qp.n];
  qp.A = new SVMFLOAT[qp.n];
  qp.b = new SVMFLOAT[qp.m];
  qp.l = new SVMFLOAT[qp.n];
  qp.u = new SVMFLOAT[qp.n];
  if(! biased){
    qp.m = 0;
  };

  which_alpha = new SVMINT[working_set_size];
  primal = new SVMFLOAT[qp.n];
  // reserve workspace for calculate_working_set
  working_set = new SVMINT[working_set_size];
  working_set_values = new SVMFLOAT[working_set_size];

  if(parameters->do_scale_y){
    epsilon_pos /= examples->get_y_var();
    epsilon_neg /= examples->get_y_var();
  };

  SVMINT i;
  //      qp.l[i] = 0 done in svm_pattern_c::
  for(i=0;i<working_set_size;i++){
    qp.l[i] = -is_zero; 
  };

  // Cpos /= (SVMFLOAT)examples_total;
  // Cneg /= (SVMFLOAT)examples_total;

  if(parameters->quadraticLossPos){
    Cpos = infinity;
  };
  if(parameters->quadraticLossNeg){
    Cneg = infinity;
  };
  //  sigfig_max = -log10(is_zero);
  lambda_WS = 0;
  to_shrink=0;

  smo.init(parameters->is_zero,parameters->convergence_epsilon,working_set_size*working_set_size);
};


void svm_c::exit_optimizer(){
  delete [](qp.c);
  delete [](qp.H);
  delete [](qp.A);
  delete [](qp.b);
  delete [](qp.l);
  delete [](qp.u);

  delete []primal;
  delete []working_set;
  delete []working_set_values;
  delete []sum;
  delete []at_bound;
  delete []which_alpha;

  primal = 0;
  working_set = 0;
  working_set_values = 0;
  sum=0;
  at_bound=0;
  which_alpha=0;
};


int svm_c::is_alpha_neg(const SVMINT i){
  // variable i is alpha*

  // take a look at svm_pattern_c::is_alpha_neg 
  // and svm_regression_c::is_alpha_neg!

  int result=0;

  if(is_pattern){
    if(all_ys[i] > 0){
      result = 1;
    }
    else{
      result = -1;
    };
  }
  else if(all_alphas[i] > 0){
    result = 1;
  }
  else if(all_alphas[i] == 0){
    result = 2*(i%2)-1;
  }
  else{
    result = -1;
  };
  return result;
};


SVMFLOAT svm_c::nabla(const SVMINT i){
  if(is_alpha_neg(i) > 0){
    return( sum[i] - all_ys[i] + epsilon_neg);
  }
  else{
    return(-sum[i] + all_ys[i] + epsilon_pos);
  };
};


SVMFLOAT svm_c::lambda(const SVMINT i){
  // size lagrangian multiplier of the active constraint

  SVMFLOAT alpha;
  SVMFLOAT result = -abs(nabla(i)+is_alpha_neg(i)*lambda_eq);
    //= -infinity; // default = not at bound

  alpha=all_alphas[i];

  if(alpha>is_zero){
    // alpha*
    if(alpha-Cneg >= - is_zero){
      // upper bound active
      result = -lambda_eq-nabla(i);
    };
  }
  else if(alpha >= -is_zero){
    // lower bound active
    if(is_alpha_neg(i) > 0){
      result = nabla(i) + lambda_eq;
    }
    else{
      result = nabla(i)-lambda_eq;
    };
  }
  else if(alpha+Cpos <= is_zero){
    // upper bound active
    result = lambda_eq - nabla(i);
  };

  return result;
};

int svm_c::feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)

  int is_feasible=1;

  //  if(at_bound[i] >= shrink_const){ is_feasible = 0; };

  SVMFLOAT alpha;

  *the_nabla = nabla(i);
  *the_lambda = lambda(i);

  alpha=all_alphas[i];

  if(alpha-Cneg >= - is_zero){
    // alpha* at upper bound
    *atbound = 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if((alpha<=is_zero) && (alpha >= -is_zero)){
    // lower bound active
    *atbound = 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if(alpha+Cpos <= is_zero){
    // alpha at upper bound
    *atbound = 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else{
    // not at bound
    *atbound = 0;
    at_bound[i] = 0;
  };
  if((*the_lambda >= feasible_epsilon) || (at_bound[i] >= shrink_const)){
    is_feasible = 0; 
  };
  return is_feasible;
};


void svm_c::set_svs(example_set_c* training_examples){
  // initialised already trained sv (for predicting or testing)

  examples = training_examples; 
  test_set = training_examples;
  examples_total = training_examples->size();
  all_alphas = examples->get_alphas();
};


void svm_c::reset_shrinked(){
  SVMINT old_ex_tot=examples_total;
  target_count=0;
  examples_total = examples->size();
  kernel->set_examples_size(examples_total);
  // unshrink, recalculate sum for all variables 
  SVMINT i,j;
  // reset all sums
  SVMFLOAT Kij;
  for(i=old_ex_tot;i<examples_total;i++){
    sum[i] = 0;
    at_bound[i] = 0;
  };
  for(i=0;i<examples_total;i++){
    if(abs(all_alphas[i])>is_zero){
      for(j=old_ex_tot;j<examples_total;j++){
	Kij = kernel->calculate_K(i,j);
	sum[j]+=all_alphas[i]*Kij;
      };
    };
  };
  sum_alpha=0;
};


SVMFLOAT svm_c::avg_norm2(){
  SVMFLOAT avg=0.0;
  for(SVMINT i=0;i<examples_total;i++){
      avg += kernel->calculate_K(i,i);
  };
  avg /= (SVMFLOAT)examples_total;
  return(avg);
};


svm_result svm_c::train(example_set_c* training_examples){
  svm_result the_result;
  examples = training_examples; 
  test_set = training_examples;
  if(parameters->verbosity>= 4){
    std::cout<<"training started"<<std::endl;
  };
  time_init = get_time();
  time_all = time_init;

  examples_total = training_examples->size();
  all_alphas = examples->get_alphas();
  all_ys = examples->get_ys();
  SVMINT param_wss = parameters->working_set_size;
  if(param_wss > examples_total){
    parameters->working_set_size = examples_total;
    working_set_size = examples_total;
  };
  if(parameters->realC <= 0.0){
    parameters->realC = 1.0/avg_norm2();
    Cpos = parameters->get_Cpos();
    Cneg = parameters->get_Cneg();
    if(parameters->verbosity>= 2){
      std::cout<<"C set to "<<parameters->realC<<std::endl;
    };
  };
  if(parameters->balance_cost){
    parameters->Lpos = (SVMFLOAT)training_examples->size_pos()/(SVMFLOAT)training_examples->size();
    parameters->Lneg = (SVMFLOAT)training_examples->size_neg()/(SVMFLOAT)training_examples->size();
    Cpos = parameters->get_Cpos();
    Cneg = parameters->get_Cneg();
  };

  // set up data structure for optimizer
  if(parameters->verbosity>= 4){
    std::cout<<"Initing optimizer"<<std::endl;
  };
  init_optimizer();

  SVMINT iteration = 0;
  SVMINT max_iterations = parameters->max_iterations;

  if(parameters->verbosity>= 4){
    std::cout<<"Initing working set"<<std::endl;
  };

  // WS-optimization
  init_working_set();

  time_init = get_time() - time_init;
  while(iteration < max_iterations){
    iteration++;
    if(parameters->verbosity>= 5){
      std::cout<<"WS is: ";
      //      SVMINT uyt;
      SVMINT ws_i;
      for(ws_i=0;ws_i<working_set_size;ws_i++){
	if(is_alpha_neg(working_set[ws_i])>0) std::cout<<"+"<<working_set[ws_i]<<" ";
	if(is_alpha_neg(working_set[ws_i])<0) std::cout<<"-"<<working_set[ws_i]<<" ";
      };
      std::cout<<std::endl;
      std::cout<<"WS alphas: ";
      for(ws_i=0;ws_i<working_set_size;ws_i++){
	std::cout<<all_alphas[working_set[ws_i]];
	if((all_alphas[working_set[ws_i]] > Cneg-is_zero) ||
	   (all_alphas[working_set[ws_i]] < -Cpos+is_zero)){
	  std::cout<<"^";
	};
	std::cout<<" ";
      };
      std::cout<<std::endl;
      std::cout<<"WS nablas: ";
      for(ws_i=0;ws_i<working_set_size;ws_i++){
	std::cout<<nabla(working_set[ws_i])<<" ";
	
      };
      std::cout<<std::endl;
      std::cout<<"WS lambdas: ";
      for(ws_i=0;ws_i<working_set_size;ws_i++){
	std::cout<<lambda(working_set[ws_i])<<" ";
	
      };
      std::cout<<std::endl;
      std::cout<<"WS at_bounds: ";
      for(ws_i=0;ws_i<working_set_size;ws_i++){
	std::cout<<at_bound[working_set[ws_i]]<<" ";
	
      };
      std::cout<<std::endl;
    };

    if(parameters->verbosity>= 4){
      std::cout<<"optimizer iteration "<<iteration<<std::endl;
    }
    else if(parameters->verbosity>=3){
      std::cout<<".";
      std::cout.flush();
    };

    optimize(); 
    put_optimizer_values();

    SVMINT i;

//     SVMFLOAT now_target=0;
//     SVMFLOAT now_target_dummy=0;
//     for(i=0;i<examples_total;i++){
// 	now_target_dummy=sum[i]/2-all_ys[i];
// 	if(is_alpha_neg(i)){
// 	    now_target_dummy+= epsilon_pos;
// 	}
// 	else{
// 	    now_target_dummy-= epsilon_neg;
// 	};
// 	now_target+=all_alphas[i]*now_target_dummy;
//     };
//     std::cout<<"Target function: "<<now_target<<std::endl;

    int conv = convergence();
    
    // old, is_zero is not touched any more
    /// @@@
    if(conv && (is_zero > parameters->is_zero)){ 
	is_zero =  parameters->is_zero;
	conv=0;
    }
    else if(conv) { 
	if(parameters->verbosity>=3){
	    // dots
	    std::cout<<std::endl;
	};
	// check convergence for all parameters
	if(0 == is_pattern){
	    for(i=0;i<examples_total;i++){
		if((all_alphas[i]<=is_zero) && (all_alphas[i]>=-is_zero)){
		    all_alphas[i]=0;
		};
	    };
	    project_to_constraint(); 
	};
	if((examples_total<examples->size()) || (target_count>0)){
	    // check convergence for all alphas  
	    if(parameters->verbosity>= 2){
		std::cout<<"***** Checking convergence for all variables"<<std::endl;
	    };
	    SVMINT old_target_count=target_count; // t_c set to 0 in reset_shrinked
	    reset_shrinked();
	    conv = convergence();
	    if(0 == conv){
		kernel->set_examples_size(examples_total);
		target_count = old_target_count;
	    };
	};
	
	if((0 == is_pattern) && (conv)){
	    // why???
	    for(i=0;i<examples_total;i++){
		if((all_alphas[i]<=is_zero) && (all_alphas[i]>=-is_zero)){
		    at_bound[i]++;
		};
	    };
	    conv = convergence();
	};
	
	if(conv){
	    if(parameters->verbosity>= 1){
		std::cout<<"*** Convergence"<<std::endl;
	    };
	    if((parameters->verbosity>=2) || 
	       (convergence_epsilon > parameters->convergence_epsilon)){
		// had to relax the KKT conditions on the way. 
		// Maybe this isn't necessary any more
		SVMFLOAT new_convergence_epsilon = 0;
		SVMFLOAT the_lambda;
		for(i=0;i<examples_total;i++){
		    the_lambda = lambda(i);
		    if(the_lambda < new_convergence_epsilon){
			new_convergence_epsilon = the_lambda;
		    };
		};
		convergence_epsilon = -new_convergence_epsilon;
	    };
	    
	    break;
	};
	
	// set variables free again
	shrink_const += 10;
	for(i=0;i<examples_total;i++){
	    at_bound[i]=0;
      };
    };

    shrink();

    calculate_working_set();
    update_working_set();

    if(parameters->verbosity >= 4){
      SVMINT shrinked=0;
      SVMINT upper_bound=0;
      SVMINT lower_bound=0;

      for(i=0;i<examples_total;i++){
	if(at_bound[i] >= shrink_const){ 
	  shrinked++;
	};
	if(abs(all_alphas[i]) < is_zero){
	  lower_bound++;
	}
	else if((all_alphas[i]-Cneg>-is_zero) || (all_alphas[i]+Cpos<is_zero)){
	  upper_bound++;
	};
      };
      std::cout<<examples_total<<" variables total, ";
      std::cout<<lower_bound<<" variables at lower bound, ";
      std::cout<<upper_bound<<" variables at upper bound, ";
      std::cout<<shrinked<<" variables shrinked"<<std::endl;
    };
  };

  SVMINT i;
  if(iteration >= max_iterations){
    std::cout<<"*** No convergence: Time up."<<std::endl;
    if(examples_total<examples->size()){
      // set sums for all variables for statistics
      reset_shrinked();
      SVMFLOAT new_convergence_epsilon = 0;
      SVMFLOAT the_lambda;
      for(i=0;i<examples_total;i++){
	the_lambda = lambda(i);
	if(the_lambda < new_convergence_epsilon){
	  new_convergence_epsilon = the_lambda;
	};
      };
      convergence_epsilon = -new_convergence_epsilon;
    };
  };

  time_all = get_time() - time_all;

  // calculate b
  SVMFLOAT new_b=0;
  SVMINT new_b_count=0;
  for(i=0;i<examples_total;i++){
    if((all_alphas[i]-Cneg < -is_zero) && (all_alphas[i]>is_zero)){
      new_b +=  all_ys[i] - sum[i]-epsilon_neg;
      new_b_count++;
    }
    else if((all_alphas[i]+Cpos > is_zero) && (all_alphas[i]<-is_zero)){
      new_b +=  all_ys[i] - sum[i]+epsilon_pos;
      new_b_count++;
    };
    examples->put_alpha(i,all_alphas[i]);
  };

  if(new_b_count>0){
    examples->put_b(new_b/((SVMFLOAT)new_b_count));
  }
  else{
    // unlikely
    for(i=0;i<examples_total;i++){
      if((all_alphas[i]<is_zero) && (all_alphas[i]>-is_zero)) {
	new_b +=  all_ys[i]- sum[i];
	new_b_count++;
      };
    };
    if(new_b_count>0){
      examples->put_b(new_b/((SVMFLOAT)new_b_count));
    }
    else{
      // even unlikelier
      for(i=0;i<examples_total;i++){
	new_b +=  all_ys[i]- sum[i];
	new_b_count++;
      };
      examples->put_b(new_b/((SVMFLOAT)new_b_count));
    };
  };


  if(parameters->verbosity>= 2){
    std::cout<<"Done training: "<<iteration<<" iterations."<<std::endl;
    if(parameters->verbosity>= 2){
      // std::cout<<"lambda_eq = "<<lambda_eq<<std::endl;
      SVMFLOAT now_target=0;
      SVMFLOAT now_target_dummy=0;
      for(i=0;i<examples_total;i++){
	now_target_dummy=sum[i]/2-all_ys[i];
	if(is_alpha_neg(i)){
	  now_target_dummy+= epsilon_pos;
	}
	else{
	  now_target_dummy-= epsilon_neg;
	};
	now_target+=all_alphas[i]*now_target_dummy;
      };
      std::cout<<"Target function: "<<now_target<<std::endl;
    };
  };

  the_result = print_statistics();

  exit_optimizer();
  examples->set_initialised_alpha();
  parameters->working_set_size = param_wss;
  return the_result;
};


void svm_c::shrink(){
  // move shrinked examples to back
  if(to_shrink>examples_total/10){
    SVMINT i;
    SVMINT last_pos=examples_total;
    if(last_pos > working_set_size){
	for(i=0;i<last_pos;i++){
	    if(at_bound[i] >= shrink_const){
		// shrinxk i
		sum_alpha += all_alphas[i];
		last_pos--;
		examples->swap(i,last_pos);
		kernel->overwrite(i,last_pos);
		sum[i] = sum[last_pos];
		at_bound[i] = at_bound[last_pos];
		if(last_pos <= working_set_size){
		    break;
		};
	    };
	};
	to_shrink=0;
	examples_total = last_pos;
	kernel->set_examples_size(examples_total);
    };
    if(parameters->verbosity>=4){
      std::cout<<"shrinked to "<<examples_total<<" variables"<<std::endl;
    };
  };
};


int svm_c::convergence(){
  long time_start = get_time();
  SVMFLOAT the_lambda_eq = 0;
  SVMINT total = 0;
  SVMFLOAT alpha_sum=0;
  SVMFLOAT alpha=0;
  SVMINT i;
  int result=1;

  // actual convergence-test
  total = 0; alpha_sum=0;

  if(biased){
    // calc lambda_eq
    for(i=0;i<examples_total;i++){
      alpha = all_alphas[i];
      alpha_sum += alpha;
      if((alpha>is_zero) && (alpha < Cneg)){ //-Cneg < -is_zero)){
	// alpha^* = - nabla
	the_lambda_eq += -nabla(i); //all_ys[i]-epsilon_neg-sum[i];
	total++;
      }
      else if((alpha<-is_zero) && (alpha > -Cpos)){ //+Cpos > is_zero)){
	// alpha = nabla
	the_lambda_eq += nabla(i); //all_ys[i]+epsilon_pos-sum[i];
	total++;
      };
    };
    
    if(parameters->verbosity >= 4){
      std::cout<<"lambda_eq = "<<(the_lambda_eq/total)<<std::endl;
    };
    if(total>0){
      lambda_eq = the_lambda_eq / total;
    }
    else{
      // keep WS lambda_eq
      lambda_eq = lambda_WS; //(lambda_eq+4*lambda_WS)/5;
      if(parameters->verbosity>= 4){
	std::cout<<"*** no SVs in convergence(), lambda_eq = "<<lambda_eq<<"."<<std::endl;
      };
    };
    
    if(target_count>2){
      // estimate lambda from WS
      if(target_count>20){
	// desperate!
	lambda_eq = ((40-target_count)*lambda_eq + (target_count-20)*lambda_WS)/20;
	if(parameters->verbosity>=5){
	  std::cout<<"Re-Re-calculated lambda from WS: "<<lambda_eq<<std::endl;
	};
	if(target_count>40){
	  // really desperate, kick one example out!
	  i = working_set[target_count%working_set_size];
	  if(is_alpha_neg(i) > 0){
	    lambda_eq = -nabla(i);
	  }
	  else{
	    lambda_eq = nabla(i);
	  };
	  if(parameters->verbosity>=5){
	    std::cout<<"set lambda_eq to nabla("<<i<<"): "<<lambda_eq<<std::endl;
	  };
	};
      }
      else{
	lambda_eq = lambda_WS;
	if(parameters->verbosity>=5){
	  std::cout<<"Re-calculated lambda_eq from WS: "<<lambda_eq<<std::endl;
	};
      };
    };

    // check linear constraint
    if(abs(alpha_sum+sum_alpha) > convergence_epsilon){
      // equality constraint violated
      project_to_constraint();
      if(parameters->verbosity>= 4){
	std::cout<<"No convergence: equality constraint violated: |"<<(alpha_sum+sum_alpha)<<"| >> 0"<<std::endl;
      };
      result = 0;  
    };
  }
  else{
    // not biased
    lambda_eq = 0.0;
  };

  i=0;
  while(i<examples_total){
    if(lambda(i)>=-convergence_epsilon){
      i++;
    }
    else{
      result = 0;
      break;
    };
  };

  time_convergence += get_time() - time_start;
  return result;
};


void svm_c::minheap_heapify(SVMINT start, SVMINT size){
  // build heap of array working_set[start:start+size-1]
  // (i.e. "size" elements starting at "start"th element)

  // minheap = 1 <=> maximal element at root 
  // (i.e. we build the heap of minimal elements)

  // v_a[i] = w_s_v[start-1+i], count beginning at 1
  SVMFLOAT* value_array = working_set_values+start-1; 
  SVMINT* pos_array = working_set+start-1;

  int running = 1;
  SVMINT pos = 1;
  SVMINT left, right, largest;
  SVMFLOAT dummyf;
  SVMINT dummyi;
  while(running){
      left = 2*pos;
      right = left+1;
      if((left<=size) && 
	 (value_array[left] > value_array[pos]))
	largest = left;
      else{
	largest = pos;
      };
      if((right<=size) && 
	 (value_array[right] > value_array[largest])){
	largest = right;
      };
      if(largest == pos){
	running = 0;
      }
      else{
	//std::cout<<"switching "<<pos<<" and "<<largest<<std::endl;
	dummyf = value_array[pos];
	dummyi = pos_array[pos];
	value_array[pos] = value_array[largest];
	pos_array[pos] = pos_array[largest];
	value_array[largest] = dummyf;
	pos_array[largest] = dummyi;
	pos = largest;
      };
  };
};


void svm_c::maxheap_heapify(SVMINT start, SVMINT size){
  // build heap of array working_set[start:start+size-1]
  // (i.e. "size" elements starting at "start"th element)

  // minheap = 1 <=> maximal element at root 
  // (i.e. we build the heap of minimal elements)

  // v_a[i] = w_s_v[start-1+i], count beginning at 1
  SVMFLOAT* value_array = working_set_values+start-1; 
  SVMINT* pos_array = working_set+start-1;

  int running = 1;
  SVMINT pos = 1;
  SVMINT left, right, largest;
  SVMFLOAT dummyf;
  SVMINT dummyi;
  while(running){
      left = 2*pos;
      right = left+1;
      if((left<=size) && 
	 (value_array[left] < value_array[pos])){
	largest = left;
      }
      else{
	largest = pos;
      };
      if((right<=size) && 
	 (value_array[right] < value_array[largest])){
	  largest = right;
      };
      if(largest == pos){
	running = 0;
      }
      else{
	dummyf = value_array[pos];
	dummyi = pos_array[pos];
	value_array[pos] = value_array[largest];
	pos_array[pos] = pos_array[largest];
	value_array[largest] = dummyf;
	pos_array[largest] = dummyi;
	pos = largest;
      };
  };
};


SVMINT svm_c::maxheap_add(SVMINT size, const SVMINT element, const SVMFLOAT value){
    if(size < (working_set_size/2+working_set_size%2)){
	// add to max_heap
	working_set_values[working_set_size/2+size] = value;
	working_set[working_set_size/2+size] = element;
	size++;
	if(size == working_set_size/2+working_set_size%2){
	    // build heap
	    SVMINT j;
	    for(j=size;j>0;j--){
		maxheap_heapify(working_set_size/2+j-1,size+1-j);
	    };
	};
    }
    else if(value >= working_set_values[working_set_size/2]){
	// replace min of max_heap
	working_set_values[working_set_size/2] = value;
	working_set[working_set_size/2] = element;
	maxheap_heapify(working_set_size/2,size);
    };
    return size;
};


SVMINT svm_c::minheap_add(SVMINT size, const SVMINT element, const SVMFLOAT value){
    if(size<working_set_size/2){
	// add to min_heap
	working_set_values[size] = value;
	working_set[size] = element;
	size++;
	if(size == working_set_size/2){
	    // build heap
	    SVMINT j;
	    for(j=size;j>0;j--){
		minheap_heapify(j-1,size+1-j);
	    };
	};
    }
    else if(value < working_set_values[0]){
	// replace max of min_heap
	working_set_values[0] = value;
	working_set[0] = element;
	minheap_heapify(0,size);
    };
    return size;
};


void svm_c::calculate_working_set(){
  /**
   *
   * Find top and bottom (w.r.t. in_alpha_neg*nabla) feasible 
   * variables for working_set
   *
   */

  long time_start = get_time();

  // reset WSS
  if(working_set_size < parameters->working_set_size){
    working_set_size=parameters->working_set_size;
    if(working_set_size>examples_total) working_set_size = examples_total;
  };

  SVMINT heap_min=0;
  SVMINT heap_max=0;

  SVMINT i=0;
  SVMFLOAT sort_value;

  working_set_values[0] = infinity;
  working_set_values[working_set_size/2] = -infinity;

  SVMFLOAT the_lambda;
  SVMFLOAT the_nabla;
  int is_feasible;
  int atbound;
  SVMINT j;

  while(i<examples_total) {
    is_feasible = feasible(i,&the_nabla,&the_lambda,&atbound);
    if(0 != is_feasible){
      if(is_alpha_neg(i) > 0){
	sort_value = -the_nabla;  // - : maximum inconsistency approach
      }
      else{
	sort_value = the_nabla;
      };
      // add to heaps
      heap_min = minheap_add(heap_min,i,sort_value);
      heap_max = maxheap_add(heap_max,i,sort_value);
    };
    i++;
  };

  if(working_set_values[0] >= working_set_values[working_set_size/2]){
    if((heap_min>0) &&
       (heap_max>0)){
      // there could be the same values in the min- and maxheap,
      // sort them out (this is very unlikely)
      j=0;
      i=0;
      while(i<heap_min){
	// working_set[i] also in max-heap?
        j=working_set_size/2;
	while((j<working_set_size/2+heap_max) && 
	      (working_set[j] != working_set[i])){
	  j++;
	};
	if(j<working_set_size/2+heap_max){
          // working_set[i] equals working_set[j]
          if(heap_min<heap_max){
            // remove j from WS
	    working_set[j] = working_set[working_set_size/2-1+heap_max];
	    heap_max--;
	  }
	  else{
	    working_set[i] = working_set[heap_min-1];
	    heap_min--;
	  };
	}
	else{
          i++;
	};
      };
    };
  };

  if(heap_min+heap_max < working_set_size) {
      // condense WS
      for(i=0;i<heap_max;i++){
	  working_set[heap_min+i] = working_set[working_set_size/2+i];
      };
      working_set_size = heap_min+heap_max;
  };

  //  if((working_set_size<examples_total) && (working_set_size>0)){
  if(target_count>0){
    // convergence error on last iteration?
    // some more tests on WS
    // unlikely to happen, so speed isn't so important

    // are all variables at the bound?
    SVMINT pos_abs;
    int bounded_pos=1;
    int bounded_neg=1;
    SVMINT pos=0;
    while((pos<working_set_size) && ((1 == bounded_pos) || (1 == bounded_neg))){
      pos_abs = working_set[pos];
      if(is_alpha_neg(pos_abs) > 0){
	if(all_alphas[pos_abs]-Cneg < -is_zero){
	  bounded_pos = 0;
	};
	if(all_alphas[pos_abs] > is_zero){
	  bounded_neg = 0;
	};
      }
      else{
	if(all_alphas[pos_abs]+Cneg > is_zero){
	  bounded_neg = 0;
	};
	if(all_alphas[pos_abs] < -is_zero){
	  bounded_pos = 0;
	};
      };
      pos++;
    };
    if(0 != bounded_pos){
      // all alphas are at upper bound
      // need alpha that can be moved upward
      // use alpha with smallest lambda
      SVMFLOAT max_lambda = infinity;
      SVMINT max_pos=examples_total;
      for(pos_abs=0;pos_abs<examples_total;pos_abs++){
	if(is_alpha_neg(pos_abs) > 0){
	  if(all_alphas[pos_abs]-Cneg < -is_zero){
	    if(lambda(pos_abs) < max_lambda){
	      max_lambda = lambda(pos_abs);
	      max_pos = pos_abs;
	    };
	  };
	}
	else{
	  if(all_alphas[pos_abs] < -is_zero){
	    if(lambda(pos_abs) < max_lambda){
	      max_lambda = lambda(pos_abs);
	      max_pos = pos_abs;
	    };
	  };
	};
      };
      if(max_pos<examples_total){
	if(working_set_size<parameters->working_set_size){
	  working_set_size++;
	};
	working_set[working_set_size-1] = max_pos;
      };
    }
    else if(0 != bounded_neg){
      // all alphas are at lower bound
      // need alpha that can be moved downward
      // use alpha with smallest lambda
      SVMFLOAT max_lambda = infinity;
      SVMINT max_pos=examples_total;
      for(pos_abs=0;pos_abs<examples_total;pos_abs++){
	if(is_alpha_neg(pos_abs) > 0){
	  if(all_alphas[pos_abs] > is_zero){
	    if(lambda(pos_abs) < max_lambda){
	      max_lambda = lambda(pos_abs);
	      max_pos = pos_abs;
	    };
	  };
	}
	else{
	  if(all_alphas[pos_abs]+Cneg > is_zero){
	    if(lambda(pos_abs) < max_lambda){
	      max_lambda = lambda(pos_abs);
	      max_pos = pos_abs;
	    };
	  };
	};
      };
      if(max_pos<examples_total){
	if(working_set_size<parameters->working_set_size){
	  working_set_size++;
	};
	working_set[working_set_size-1] = max_pos;
      };
    };
  };

  if((working_set_size<parameters->working_set_size) &&
     (working_set_size<examples_total)){
    // use full working set
    SVMINT pos = (SVMINT)((SVMFLOAT)examples_total*rand()/(RAND_MAX+1.0));
    int ok;
    while((working_set_size<parameters->working_set_size) &&
	  (working_set_size<examples_total)){
      // add pos into WS if it isn't already
      ok = 1;
      for(i=0;i<working_set_size;i++){
	if(working_set[i] == pos){
	  ok=0;
	  i = working_set_size;
	};
      };
      if(1 == ok){
	working_set[working_set_size] = pos;
	working_set_size++;
      };
      pos = (pos+1)%examples_total;
    };
  };

  SVMINT ipos;
  for(ipos=0;ipos<working_set_size;ipos++){
    which_alpha[ipos] = is_alpha_neg(working_set[ipos]);
  };

  time_calc += get_time() - time_start;
  return;
};


void svm_c::project_to_constraint(){
  // project alphas to match the constraint
  if(biased){
    SVMFLOAT alpha_sum = sum_alpha;
    SVMINT SVcount=0;
    SVMFLOAT alpha;
    SVMINT i;
    for(i=0;i<examples_total;i++){
      alpha = all_alphas[i];
      alpha_sum += alpha;
      if(((alpha>is_zero) && (alpha-Cneg < -is_zero)) ||
	 ((alpha<-is_zero) && (alpha+Cpos > is_zero))){
	SVcount++;
      };
    };
    if(SVcount > 0){
      // project
      alpha_sum /= (SVMFLOAT)SVcount;
      for(i=0;i<examples_total;i++){
	alpha = all_alphas[i];
	if(((alpha>is_zero) && (alpha-Cneg < -is_zero)) ||
	   ((alpha<-is_zero) && (alpha+Cpos > is_zero))){
	  all_alphas[i] -= alpha_sum;
	};
      };
    };
  };
};


void svm_c::init_working_set(){
  // calculate sum
  SVMINT i,j;

  project_to_constraint();
  // check bounds!
  if(examples->initialised_alpha()){
    if(parameters->verbosity >= 2){
      std::cout<<"Initialising variables, this may take some time."<<std::endl;
    };
    for(i=0; i<examples_total;i++){
      sum[i] = 0;
      at_bound[i] = 0;
      for(j=0; j<examples_total;j++){
	sum[i] += all_alphas[j]*kernel->calculate_K(i,j);
      };
    };
  }
  else{
    // skip kernel calculation as all alphas = 0
    for(i=0; i<examples_total;i++){
      sum[i] = 0;
      at_bound[i] = 0;
    };    
  };

  if(examples->initialised_alpha()){
    calculate_working_set();
  }
  else{
    // first working set is random
    j=0;
    i=0;
    while((i<working_set_size) && (j < examples_total)){
      working_set[i] = j;
      if(is_alpha_neg(j) > 0){
	which_alpha[i] = 1;
      }
      else{
	which_alpha[i] = -1;
      };
      i++;
      j++;
    };
  };   
  update_working_set();
};


void svm_c::put_optimizer_values(){
  // update nabla, sum, examples.
  // sum[i] += (primal_j^*-primal_j-alpha_j^*+alpha_j)K(i,j)
  // check for |nabla| < is_zero (nabla <-> nabla*)
  //  std::cout<<"put_optimizer_values()"<<std::endl;
  SVMINT i=0; 
  SVMINT j=0;
  SVMINT pos_i;
  SVMFLOAT the_new_alpha;
  SVMFLOAT* kernel_row;
  SVMFLOAT alpha_diff;

  long time_start = get_time();
  pos_i=working_set_size;
  while(pos_i>0){
    pos_i--;
    if(which_alpha[pos_i]>0){
      the_new_alpha = primal[pos_i];
    }
    else{
      the_new_alpha = -primal[pos_i];
    };
    // next three statements: keep this order!
    i = working_set[pos_i];
    alpha_diff = the_new_alpha-all_alphas[i];
    all_alphas[i] = the_new_alpha;

    if(alpha_diff != 0){
      // update sum ( => nabla)
      kernel_row = kernel->get_row(i);
      for(j=0;j<examples_total;j++){
	sum[j] += alpha_diff*kernel_row[j];
      };
    };
  };
  time_update += get_time() - time_start;
};


void svm_c::update_working_set(){
  long time_start = get_time();
  // setup subproblem
  SVMINT i,j;
  SVMINT pos_i, pos_j;
  SVMFLOAT* kernel_row;
  SVMFLOAT sum_WS;

  for(pos_i=0;pos_i<working_set_size;pos_i++){
    i = working_set[pos_i];

    // put row sort_i in hessian 
    kernel_row = kernel->get_row(i);
    sum_WS=0;
    //    for(pos_j=0;pos_j<working_set_size;pos_j++){
    for(pos_j=0;pos_j<pos_i;pos_j++){
      j = working_set[pos_j];
      // put all elements K(i,j) in hessian, where j in WS
      if(((which_alpha[pos_j] < 0) && (which_alpha[pos_i] < 0)) ||
	 ((which_alpha[pos_j] > 0) && (which_alpha[pos_i] > 0))){
	// both i and j positive or negative
	(qp.H)[pos_i*working_set_size+pos_j] = kernel_row[j];
	(qp.H)[pos_j*working_set_size+pos_i] = kernel_row[j];
      }
      else{
	// one of i and j positive, one negative
	(qp.H)[pos_i*working_set_size+pos_j] = -kernel_row[j];
	(qp.H)[pos_j*working_set_size+pos_i] = -kernel_row[j];
      };
    };
    for(pos_j=0;pos_j<working_set_size;pos_j++){
      j = working_set[pos_j];
      sum_WS+=all_alphas[j]*kernel_row[j];
    };
    // set main diagonal 
    (qp.H)[pos_i*working_set_size+pos_i] = kernel_row[i];

    // linear and box constraints
    if(which_alpha[pos_i]<0){
      // alpha
      (qp.A)[pos_i] = -1;
      // lin(alpha) = y_i+eps-sum_{i not in WS} alpha_i K_{ij}
      //            = y_i+eps-sum_i+sum_{i in WS}
      (qp.c)[pos_i] = all_ys[i]+epsilon_pos-sum[i]+sum_WS;
      primal[pos_i] = -all_alphas[i];
      (qp.u)[pos_i] = Cpos;
    }
    else{
      // alpha^*
      (qp.A)[pos_i] = 1;
      (qp.c)[pos_i] = -all_ys[i]+epsilon_neg+sum[i]-sum_WS;
      primal[pos_i] = all_alphas[i];
      (qp.u)[pos_i] = Cneg;
    };
  };
  if(parameters->quadraticLossNeg){
    for(pos_i=0;pos_i<working_set_size;pos_i++){
      if(which_alpha[pos_i]>0){
	(qp.H)[pos_i*(working_set_size+1)] += 1/Cneg;
	(qp.u)[pos_i] = infinity;
      };
    };
  };
  if(parameters->quadraticLossPos){
    for(pos_i=0;pos_i<working_set_size;pos_i++){
      if(which_alpha[pos_i]<0){
	(qp.H)[pos_i*(working_set_size+1)] += 1/Cpos;
	(qp.u)[pos_i] = infinity;
      };
    };
  };

  time_update += get_time() - time_start; 
};


svm_result svm_c::test(example_set_c* test_examples, int verbose){
  svm_result the_result;
  test_set = test_examples;

  SVMINT i;
  SVMFLOAT MAE=0;
  SVMFLOAT MSE=0;
  SVMFLOAT actloss=0;
  SVMFLOAT theloss=0;
  SVMFLOAT theloss_pos=0;
  SVMFLOAT theloss_neg=0;
  SVMINT countpos=0;
  SVMINT countneg=0;
  // for pattern:
  SVMINT correct_pos=0;
  SVMINT correct_neg=0;
  SVMINT total_pos=0;
  SVMINT total_neg=0;

  SVMFLOAT prediction;
  SVMFLOAT y;
  svm_example example;
  for(i=0;i<test_set->size();i++){
    example = test_set->get_example(i);
    prediction = predict(example);
    y = examples->unscale_y(test_set->get_y(i));
    MAE += abs(y-prediction);
    MSE += (y-prediction)*(y-prediction);
    actloss=loss(prediction,y);
    theloss+=actloss;
    if(y < prediction-parameters->epsilon_pos){
      theloss_pos += actloss;
      countpos++;
    }
    else if(y > prediction+parameters->epsilon_neg){
      theloss_neg += actloss;
      countneg++;
    };
    // if pattern!
    if(is_pattern){
      if(y>0){
	if(prediction>0){
	  correct_pos++;
	};
	total_pos++;
      }
      else{
	if(prediction<=0){
	  correct_neg++;
	};
	total_neg++;
      };
    };    
  };
  if(countpos != 0){
    theloss_pos /= (SVMFLOAT)countpos;
  };
  if(countneg != 0){
    theloss_neg /= (SVMFLOAT)countneg;
  };

  the_result.MAE =  MAE / (SVMFLOAT)test_set->size();
  the_result.MSE =  MSE / (SVMFLOAT)test_set->size();
  the_result.loss = theloss/test_set->size();
  the_result.loss_pos = theloss_pos;
  the_result.loss_neg = theloss_neg;
  the_result.number_svs = 0;
  the_result.number_bsv = 0;
  if(is_pattern){
    the_result.accuracy = ((SVMFLOAT)(correct_pos+correct_neg))/((SVMFLOAT)(total_pos+total_neg));
    the_result.precision = ((SVMFLOAT)correct_pos/((SVMFLOAT)(correct_pos+total_neg-correct_neg)));
    the_result.recall = ((SVMFLOAT)correct_pos/(SVMFLOAT)total_pos);
  }
  else{
    the_result.accuracy = -1;
    the_result.precision = -1;
    the_result.recall = -1;
  };

  if(verbose){
    std::cout << "Average loss  : "<<(theloss/test_set->size())<<std::endl;
    std::cout << "Avg. loss pos : "<<theloss_pos<<"\t ("<<countpos<<" occurences)"<<std::endl;
    std::cout << "Avg. loss neg : "<<theloss_neg<<"\t ("<<countneg<<" occurences)"<<std::endl;
    std::cout << "Mean absolute error : "<<the_result.MAE<<std::endl;
    std::cout << "Mean squared error  : "<<the_result.MSE<<std::endl;

    if(is_pattern){
      // output precision, recall and accuracy
      std::cout<<"Accuracy  : "<<the_result.accuracy<<std::endl;
      std::cout<<"Precision : "<<the_result.precision<<std::endl;
      std::cout<<"Recall    : "<<the_result.recall<<std::endl;
      // nice printout ;-)
      int rows = (int)(1+log10((SVMFLOAT)(total_pos+total_neg)));
      int now_digits = rows+2;
      int i,j;
      std::cout<<std::endl;
      std::cout<<"Predicted values:"<<std::endl;
      std::cout<<"   |";
      for(i=0;i<rows;i++){ std::cout<<" "; };
      std::cout<<"+  |";
      for(j=0;j<rows;j++){ std::cout<<" "; };
      std::cout<<"-"<<std::endl;
      
      std::cout<<"---+";
      for(i=0;i<now_digits;i++){ std::cout<<"-"; };
      std::cout<<"-+-";
      for(i=0;i<now_digits;i++){ std::cout<<"-"; };
      std::cout<<std::endl;
      
      std::cout<<" + |  ";
      now_digits=rows-(int)(1+log10((SVMFLOAT)correct_pos))-1;
      for(i=0;i<now_digits;i++){ std::cout<<" "; };
      std::cout<<correct_pos<<"  |  ";
      now_digits=rows-(int)(1+log10((SVMFLOAT)(total_pos-correct_pos)))-1;
      for(i=0;i<now_digits;i++){ std::cout<<" "; };
      std::cout<<total_pos-correct_pos<<"    (true pos)"<<std::endl;
      
      std::cout<<" - |  ";
      now_digits=rows-(int)(1+log10((SVMFLOAT)(total_neg-correct_neg)))-1;
      for(i=0;i<now_digits;i++){ std::cout<<" "; };
      std::cout<<(total_neg-correct_neg)<<"  |  ";
      now_digits=rows-(int)(1+log10((SVMFLOAT)correct_neg))-1;
      for(i=0;i<now_digits;i++){ std::cout<<" "; };
      std::cout<<correct_neg<<"    (true neg)"<<std::endl;
      std::cout<<std::endl;
    };
  };
  return the_result;
};


void svm_c::optimize(){
  // optimizer-specific call
  // get time
  long time_start = get_time();

  qp.n = working_set_size;

  SVMINT i;
  SVMINT j;

  // equality constraint
  qp.b[0]=0;
  for(i=0;i<working_set_size;i++){
    qp.b[0] += all_alphas[working_set[i]];
  };

  // set initial optimization parameters
  SVMFLOAT new_target=0;
  SVMFLOAT old_target=0;
  SVMFLOAT target_tmp;
  for(i=0;i<working_set_size;i++){
    target_tmp = primal[i]*qp.H[i*working_set_size+i]/2;
    for(j=0;j<i;j++){
      target_tmp+=primal[j]*qp.H[j*working_set_size+i];
    };
    target_tmp+=qp.c[i];
    old_target+=target_tmp*primal[i];
  };

  SVMFLOAT new_constraint_sum=0;
  SVMFLOAT my_is_zero = is_zero;
  SVMINT sv_count=working_set_size;

  qp.n = working_set_size;
  // optimize
  int KKTerror=1;
  int convError=0;

  smo.set_max_allowed_error(convergence_epsilon);

  // loop while some KKT condition is not valid (alpha=0)

  int result;
  if(biased){
    result = smo.smo_solve(&qp,primal);
    lambda_WS = smo.get_lambda_eq();
  }
  else{
    result = smo.smo_solve_single(&qp,primal);
    lambda_WS = 0.0;
  };

  /////////// new
  SVMINT it=3;
  if(! is_pattern){
      // iterate optimization 3 times with changed sign on variables, if KKT conditions are not satisfied
    SVMFLOAT lambda_lo;
    while(KKTerror && (it>0)){
      KKTerror = 0;
      it--;
      for(i=0;i<working_set_size;i++){
	if(primal[i]<is_zero){
	  lambda_lo =  epsilon_neg + epsilon_pos - qp.c[i];
	  for(j=0;j<working_set_size;j++){
	    lambda_lo -= primal[j]*qp.H[i*working_set_size+j];
	  };
	  if(qp.A[i] > 0){
	    lambda_lo -= lambda_WS;
	  }
	  else{
	    lambda_lo += lambda_WS;
	  };

	  if(lambda_lo<-convergence_epsilon){
	    // change sign of i
	    KKTerror=1;
	    qp.A[i] = -qp.A[i];
	    which_alpha[i] = -which_alpha[i];
	    primal[i] = -primal[i];
	    qp.c[i] = epsilon_neg + epsilon_pos - qp.c[i];
	    if(qp.A[i]>0){
	      qp.u[i] = Cneg;
	    }
	    else{
	      qp.u[i] = Cpos;
	    };
	    for(j=0;j<working_set_size;j++){
	      qp.H[i*working_set_size+j] = -qp.H[i*working_set_size+j];
	      qp.H[j*working_set_size+i] = -qp.H[j*working_set_size+i];
	    };
	    if(parameters->quadraticLossNeg){
	      if(which_alpha[i]>0){
		(qp.H)[i*(working_set_size+1)] += 1/Cneg;
		(qp.u)[i] = infinity;
	      }
	      else{
		// previous was neg
		(qp.H)[i*(working_set_size+1)] -= 1/Cneg;
	      };
	    };
	    if(parameters->quadraticLossPos){
	      if(which_alpha[i]<0){
		(qp.H)[i*(working_set_size+1)] += 1/Cpos;
		(qp.u)[i] = infinity;
	      }
	      else{
		//previous was pos
		(qp.H)[i*(working_set_size+1)] -= 1/Cpos;
	      };
	    };
	  };
	};
      };
      result = smo.smo_solve(&qp,primal);
      if(biased){
	lambda_WS = smo.get_lambda_eq();
      }; // sonst 0 beibehalten
    };
  };

  KKTerror = 1;
  //////////////////////

  if(parameters->verbosity>=5){
    std::cout<<"smo ended with result "<<result<<std::endl;
    std::cout<<"lambda_WS = "<<lambda_WS<<std::endl;
    std::cout<<"smo: Resulting values:"<<std::endl;
    for(i=0;i<working_set_size;i++){
      std::cout<<i<<": "<<primal[i]<<std::endl; 
    };
  };

  while(KKTerror){
    // clip
    sv_count=working_set_size;
    new_constraint_sum=qp.b[0];
    if(biased){
      // more checks for feasibility
      for(i=0;i<working_set_size;i++){
	// check if at bound
	if(primal[i] <= my_is_zero){
	  // at lower bound
	  primal[i] = qp.l[i];
	  sv_count--;
	}
	else if(qp.u[i]-primal[i] <= my_is_zero){
	  // at upper bound
	  primal[i] = qp.u[i];
	  sv_count--;
	};
	new_constraint_sum -= qp.A[i]*primal[i];
      };
      
      // enforce equality constraint
      if(sv_count>0){
	new_constraint_sum /= (SVMFLOAT)sv_count;
	if(parameters->verbosity>=5){
	  std::cout<<"adjusting "<<sv_count<<" alphas by "<<new_constraint_sum<<std::endl;
	};
	for(i=0;i<working_set_size;i++){
	  if((primal[i] > qp.l[i]) && 
	     (primal[i] < qp.u[i])){
	    // real sv
	    primal[i] += qp.A[i]*new_constraint_sum;
	  };
	};
      }
      else if(abs(new_constraint_sum)>(SVMFLOAT)working_set_size*is_zero){
	// error, can't get feasible point
	if(parameters->verbosity>=5){
	  std::cout<<"WARNING: No SVs, constraint_sum = "<<new_constraint_sum<<std::endl;
	};
	old_target = -infinity; 
	//is_ok=0;
	convError=1;
      };
    };
    // test descend
    new_target=0;
    for(i=0;i<working_set_size;i++){
      // attention: loqo changes one triangle of H!
      target_tmp = primal[i]*qp.H[i*working_set_size+i]/2;
      for(j=0;j<i;j++){
	target_tmp+=primal[j]*qp.H[j*working_set_size+i];
      };
      target_tmp+=qp.c[i];
      new_target+=target_tmp*primal[i];
    };

    if(new_target < old_target){
      KKTerror = 0;
      if(parameters->descend < old_target - new_target){
	target_count=0;
      }
      else{
	convError=1;
      };
      if(parameters->verbosity>=5){
	std::cout<<"descend = "<<old_target-new_target<<std::endl;
      };
    }
    else if(sv_count > 0){
      // less SVs
      // set my_is_zero to min_i(primal[i]-qp.l[i], qp.u[i]-primal[i])
      my_is_zero = 1e20;
      for(i=0;i<working_set_size;i++){
	if((primal[i] > qp.l[i]) && (primal[i] < qp.u[i])){
	  if(primal[i] - qp.l[i] < my_is_zero){
	    my_is_zero = primal[i]-qp.l[i];
	  };
	  if(qp.u[i]  - primal[i]  < my_is_zero){
	    my_is_zero = qp.u[i] - primal[i];
	  };
	};
      };
      if(target_count == 0){
      	my_is_zero *= 2;
      };
      if(parameters->verbosity>=5){
	std::cout<<"WARNING: no descend ("<<old_target-new_target
	    <<" <= "<<parameters->descend
	  //	    <<", alpha_diff = "<<alpha_diff
	    <<"), adjusting is_zero to "<<my_is_zero<<std::endl;
	std::cout<<"new_target = "<<new_target<<std::endl;
      };
    }
    else{
      // nothing we can do
      if(parameters->verbosity>=5){
	std::cout<<"WARNING: no descend ("<<old_target-new_target
	    <<" <= "<<parameters->descend<<"), stopping."<<std::endl;
      };
      KKTerror=0;
      convError=1;
    };
  };

  if(1 == convError){
    target_count++;
    //    sigfig_max+=0.05;
    if(old_target < new_target){
      for(i=0;i<working_set_size;i++){
	primal[i] = qp.A[i]*all_alphas[working_set[i]];
      };                              
      if(parameters->verbosity>=5){	
	std::cout<<"WARNING: Convergence error, restoring old primals"<<std::endl; 
      };
    };                                          
  };

  if(target_count>50){
    // non-recoverable numerical error
    convergence_epsilon*=2;
    feasible_epsilon = convergence_epsilon;
    //    sigfig_max=-log10(is_zero);
    if(parameters->verbosity>=1)
      std::cout<<"WARNING: reducing KKT precision to "<<convergence_epsilon<<std::endl;
    target_count=0;
  };

  time_optimize += get_time() - time_start;
};


void svm_c::predict(example_set_c* test_examples){
  test_set = test_examples;
  SVMINT i;
  SVMFLOAT prediction;
  svm_example example;

  for(i=0;i<test_set->size();i++){
    example = test_set->get_example(i);
    prediction = predict(example);
    test_set->put_y(i,prediction);
  };
  test_set->set_initialised_y();
  test_set->put_b(examples->get_b());
  if(parameters->verbosity>=4){
    std::cout<<"Prediction generated"<<std::endl;
  };
};


SVMFLOAT svm_c::predict(svm_example example){ 
  SVMINT i;
  svm_example sv;
  SVMFLOAT the_sum=examples->get_b();

  for(i=0;i<examples_total;i++){
    if(all_alphas[i] != 0){
      sv = examples->get_example(i);
      the_sum += all_alphas[i]*kernel->calculate_K(sv,example);
    };
  };
  the_sum = examples->unscale_y(the_sum);
  if(parameters->use_min_prediction){
    if(the_sum < parameters->min_prediction){
      the_sum = parameters->min_prediction;
    };
  };
  return the_sum;
};


SVMFLOAT svm_c::predict(SVMINT i){
  //   return (sum[i]+examples->get_b());
  // numerically more stable:
  return predict(examples->get_example(i));
};

SVMFLOAT svm_c::loss(SVMINT i){
  return loss(predict(i),examples->unscale_y(all_ys[i]));
};


SVMFLOAT svm_c::loss(SVMFLOAT prediction, SVMFLOAT value){
  SVMFLOAT theloss = prediction - value;
  if(is_pattern){
    if(((value > 0) && (prediction > 0)) ||
       ((value <= 0) && (prediction <= 0))){
      theloss = 0;
    }
  };
  if(theloss > parameters->epsilon_pos){ 
    if(parameters->quadraticLossPos){
      theloss = parameters->Lpos*(theloss-parameters->epsilon_pos)
	*(theloss-parameters->epsilon_pos); 
    }
    else{
      theloss =  parameters->Lpos*(theloss-parameters->epsilon_pos); 
    };
  }
  else if(theloss >= -parameters->epsilon_neg){ theloss = 0; }
  else{ 
    if(parameters->quadraticLossNeg){
      theloss = parameters->Lneg*(-theloss-parameters->epsilon_neg)
	*(-theloss-parameters->epsilon_neg);
    }
    else{
      theloss = parameters->Lneg*(-theloss-parameters->epsilon_neg);
    };
  };
  return theloss;
};


void svm_c::print_special_statistics(){
  // nothing special here!
};


svm_result svm_c::print_statistics(){
  // # SV, # BSV, pos&neg, Loss, VCdim
  // Pattern: Acc, Rec, Pred

  if(parameters->verbosity>=2){
    std::cout<<"----------------------------------------"<<std::endl;
  };

  svm_result the_result;
  if(test_set->size() <= 0){
    if(parameters->verbosity>= 0){
      std::cout << "No training set given" << std::endl;
    };
    the_result.loss = -1;
    return the_result;  // undefined
  };
  SVMINT i;
  SVMINT svs = 0;
  SVMINT bsv = 0;
  SVMFLOAT actloss = 0;
  SVMFLOAT theloss = 0;
  SVMFLOAT theloss_pos=0;
  SVMFLOAT theloss_neg=0;
  SVMINT countpos=0;
  SVMINT countneg=0;
  SVMFLOAT min_alpha=infinity;
  SVMFLOAT max_alpha=-infinity;
  SVMFLOAT norm_w=0;
  SVMFLOAT max_norm_x=0;
  SVMFLOAT min_norm_x=1e20;
  SVMFLOAT norm_x=0;
  SVMFLOAT loo_loss_estim=0;
  // for pattern:
  SVMFLOAT correct_pos=0;
  SVMINT correct_neg=0;
  SVMINT total_pos=0;
  SVMINT total_neg=0;
  SVMINT estim_pos=0;
  SVMINT estim_neg=0;
  SVMFLOAT MSE=0;
  SVMFLOAT MAE=0;
  SVMFLOAT alpha;
  SVMFLOAT prediction;
  SVMFLOAT y;
  SVMFLOAT xi;

  for(i=0;i<examples_total;i++){
    // needed before test-loop for performance estimators
    norm_w+=all_alphas[i]*sum[i];

    alpha=all_alphas[i];
    if(alpha!=0){
      norm_x = kernel->calculate_K(i,i);
      if(norm_x>max_norm_x){
	max_norm_x = norm_x;
      };
      if(norm_x<min_norm_x){
	min_norm_x = norm_x;
      };
    };
  };
  
  SVMFLOAT r_delta = max_norm_x;
  if(parameters->loo_estim){
    r_delta = 0;
    SVMFLOAT r_current;
    for(SVMINT j=0;j<examples_total;j++){
      norm_x = kernel->calculate_K(j,j);
      for(i=0;i<examples_total;i++){
	r_current = norm_x-kernel->calculate_K(i,j);
	if(r_current > r_delta){
	  r_delta = r_current;
	};
      };
    };
  };

  for(i=0;i<examples_total;i++){
    alpha=all_alphas[i];
    if(alpha<min_alpha) min_alpha = alpha;
    if(alpha>max_alpha) max_alpha = alpha;
    prediction = predict(i);
    y = examples->unscale_y(all_ys[i]);
    actloss=loss(prediction,y);
    theloss+=actloss;
    MAE += abs(prediction-y);
    MSE += (prediction-y)*(prediction-y);
    if(y < prediction-parameters->epsilon_pos){
      theloss_pos += actloss;
      countpos++;
    }
    else if(y > prediction+parameters->epsilon_neg){
      theloss_neg += actloss;
      countneg++;
    };
    if(parameters->loo_estim){
      if(abs(alpha)>is_zero){ 
	if(is_alpha_neg(i)>=0){
	  loo_loss_estim += loss(prediction-(abs(alpha)*(2*kernel->calculate_K(i,i)+r_delta)+2*epsilon_neg),y);
	}
	else{
	  loo_loss_estim += loss(prediction+(abs(alpha)*(2*kernel->calculate_K(i,i)+r_delta)+2*epsilon_pos),y);
	};
      };
    }
    else{
      // loss doesn't change if non-SV is omitted
      loo_loss_estim += actloss;
    };

    if(abs(alpha)>is_zero){ 
     // a support vector
      svs++; 
      if((alpha-Cneg >= -is_zero) || (alpha+Cpos <= is_zero)){ 
	bsv++; 
      };
    };

    if(is_pattern){
      if(y>0){
	if(prediction>0){
	  correct_pos++;
	};
	if(prediction>1){
	  xi=0;
	}
	else{
	  xi=1-prediction;
	};
	if(2*alpha*r_delta+xi >= 1){
	  estim_pos++;
	};
	total_pos++;
      }
      else{
	if(prediction<=0){
	  correct_neg++;
	};
	if(prediction<-1){
	  xi=0;
	}
	else{
	  xi=1+prediction;
	};
	if(2*(-alpha)*r_delta+xi >= 1){
	  estim_neg++;
	};
	total_neg++;
      };
    };    
  };
  if(countpos != 0){
    theloss_pos /= (SVMFLOAT)countpos;
  };
  if(countneg != 0){
    theloss_neg /= (SVMFLOAT)countneg;
  };

  the_result.MAE = MAE / (SVMFLOAT)examples_total;
  the_result.MSE = MSE / (SVMFLOAT)examples_total;
  the_result.VCdim = 1+norm_w*max_norm_x;
  the_result.loss = theloss/((SVMFLOAT)examples_total);
  if(parameters->loo_estim){
    the_result.pred_loss = loo_loss_estim/((SVMFLOAT)examples_total);
  }
  else{
    the_result.pred_loss = the_result.loss;
  };
  the_result.loss_pos = theloss_pos;
  the_result.loss_neg = theloss_neg;
  the_result.number_svs = svs;
  the_result.number_bsv = bsv;
  if(is_pattern){
    the_result.accuracy = ((SVMFLOAT)(correct_pos+correct_neg))/((SVMFLOAT)(total_pos+total_neg));
    the_result.precision = ((SVMFLOAT)correct_pos/((SVMFLOAT)(correct_pos+total_neg-correct_neg)));
    the_result.recall = ((SVMFLOAT)correct_pos/(SVMFLOAT)total_pos);
    if(parameters->loo_estim){
      the_result.pred_accuracy = (1-((SVMFLOAT)(estim_pos+estim_neg))/((SVMFLOAT)(total_pos+total_neg)));
      the_result.pred_precision = ((SVMFLOAT)(total_pos-estim_pos))/((SVMFLOAT)(total_pos-estim_pos+estim_neg));
      the_result.pred_recall = (1-(SVMFLOAT)estim_pos/((SVMFLOAT)total_pos));
    }
    else{
      the_result.pred_accuracy = the_result.accuracy;
      the_result.pred_precision = the_result.precision;
      the_result.pred_recall = the_result.recall;
    };
  }
  else{
    the_result.accuracy = -1;
    the_result.precision = -1;
    the_result.recall = -1;
    the_result.pred_accuracy = -1;
    the_result.pred_precision = -1;
    the_result.pred_recall = -1;
  };


  if(convergence_epsilon > parameters->convergence_epsilon){
    std::cout<<"WARNING: The results were obtained using a relaxed epsilon of "<<convergence_epsilon<<" on the KKT conditions!"<<std::endl;
  }
  else if(parameters->verbosity>=2){
    std::cout<<"The results are valid with an epsilon of "<<convergence_epsilon<<" on the KKT conditions."<<std::endl;
  };
  if(parameters->verbosity >= 2){
    std::cout << "Average loss  : "<<the_result.loss<<" (loo-estim: "<< the_result.pred_loss<<")"<<std::endl;
    std::cout << "Avg. loss pos : "<<theloss_pos<<"\t ("<<countpos<<" occurences)"<<std::endl;
    std::cout << "Avg. loss neg : "<<theloss_neg<<"\t ("<<countneg<<" occurences)"<<std::endl;
    std::cout << "Mean absolute error : "<<the_result.MAE<<std::endl;
    std::cout << "Mean squared error  : "<<the_result.MSE<<std::endl;
    std::cout << "Support Vectors : "<<svs<<std::endl;
    std::cout << "Bounded SVs     : "<<bsv<<std::endl;
    std::cout<<"min SV: "<<min_alpha<<std::endl
	<<"max SV: "<<max_alpha<<std::endl;
    std::cout<<"|w| = "<<sqrt(norm_w)<<std::endl;
    std::cout<<"max |x| = "<<sqrt(max_norm_x)<<std::endl;
    std::cout<<"VCdim <= "<<the_result.VCdim<<std::endl;

    print_special_statistics();

    if((is_pattern) && (! parameters->is_distribution)){
      // output precision, recall and accuracy
      if(parameters->loo_estim){
	std::cout<<"performance (+estimators):"<<std::endl;
	std::cout<<"Accuracy  : "<<the_result.accuracy<<" ("<<the_result.pred_accuracy<<")"<<std::endl;
	std::cout<<"Precision : "<<the_result.precision<<" ("<<the_result.pred_precision<<")"<<std::endl;
	std::cout<<"Recall    : "<<the_result.recall<<" ("<<the_result.pred_recall<<")"<<std::endl;
      }
      else{
	std::cout<<"performance :"<<std::endl;
	std::cout<<"Accuracy  : "<<the_result.accuracy<<std::endl;
	std::cout<<"Precision : "<<the_result.precision<<std::endl;
	std::cout<<"Recall    : "<<the_result.recall<<std::endl;
      };
      if(parameters->verbosity>= 2){
	// nice printout ;-)
	int rows = (int)(1+log10((SVMFLOAT)(total_pos+total_neg)));
	int now_digits = rows+2;
	int i,j;
	std::cout<<std::endl;
	std::cout<<"Predicted values:"<<std::endl;
	std::cout<<"   |";
	for(i=0;i<rows;i++){ std::cout<<" "; };
	std::cout<<"+  |";
	for(j=0;j<rows;j++){ std::cout<<" "; };
	std::cout<<"-"<<std::endl;

	std::cout<<"---+";
	for(i=0;i<now_digits;i++){ std::cout<<"-"; };
	std::cout<<"-+-";
	for(i=0;i<now_digits;i++){ std::cout<<"-"; };
	std::cout<<std::endl;

	std::cout<<" + |  ";
	now_digits=rows-(int)(1+log10((SVMFLOAT)correct_pos))-1;
	for(i=0;i<now_digits;i++){ std::cout<<" "; };
	std::cout<<correct_pos<<"  |  ";
	now_digits=rows-(int)(1+log10((SVMFLOAT)(total_pos-correct_pos)))-1;
	for(i=0;i<now_digits;i++){ std::cout<<" "; };
	std::cout<<total_pos-correct_pos<<"    (true pos)"<<std::endl;

	std::cout<<" - |  ";
	now_digits=rows-(int)(1+log10((SVMFLOAT)(total_neg-correct_neg)))-1;
	for(i=0;i<now_digits;i++){ std::cout<<" "; };
	std::cout<<(total_neg-correct_neg)<<"  |  ";
	now_digits=rows-(int)(1+log10((SVMFLOAT)correct_neg))-1;
	for(i=0;i<now_digits;i++){ std::cout<<" "; };
	std::cout<<correct_neg<<"    (true neg)"<<std::endl;
	std::cout<<std::endl;
      };
    };
  };

  SVMINT dim = examples->get_dim();
  if(((parameters->print_w == 1) && (parameters->is_linear != 0)) ||
     ((dim<100) && 
      (parameters->verbosity>= 2) && 
      (parameters->is_linear != 0))
     ){
    // print hyperplane
    SVMINT j;
    svm_example example;
    SVMFLOAT* w = new SVMFLOAT[dim];
    SVMFLOAT b = examples->get_b();
    for(j=0;j<dim;j++) w[j] = 0;
    for(i=0;i<examples_total;i++){
      example = examples->get_example(i);
      alpha = examples->get_alpha(i);
      for(j=0;j<example.length;j++){
	w[((example.example)[j]).index] += alpha*((example.example)[j]).att;
      };
    };
    if(examples->initialised_scale()){
      SVMFLOAT* exp = examples->get_exp();
      SVMFLOAT* var = examples->get_var();
      for(j=0;j<dim;j++){
	if(var[j] != 0){
	  w[j] /= var[j];
	};
	if(0 != var[dim]){
	  w[j] *= var[dim];
	};
	b -= w[j]*exp[j];
      };
      b += exp[dim];
    };
    for(j=0;j<dim;j++){
      std::cout << "w["<<j<<"] = " << w[j] << std::endl;
    };
    std::cout << "b = "<<b<<std::endl;
    if(dim==1){
      std::cout<<"y = "<<w[0]<<"*x+"<<b<<std::endl;
    };
    if((dim==2) && (is_pattern)){
      std::cout<<"x1 = "<<-w[0]/w[1]<<"*x0+"<<-b/w[1]<<std::endl;
    };
    delete []w;
  };

  if(parameters->verbosity>= 2){
    std::cout<<"Time for learning:"<<std::endl
	<<"init        : "<<(time_init/100)<<"s"<<std::endl
	<<"optimizer   : "<<(time_optimize/100)<<"s"<<std::endl
	<<"convergence : "<<(time_convergence/100)<<"s"<<std::endl
	<<"update ws   : "<<(time_update/100)<<"s"<<std::endl
	<<"calc ws     : "<<(time_calc/100)<<"s"<<std::endl
	<<"============="<<std::endl
	<<"all         : "<<(time_all/100)<<"s"<<std::endl;
  }
  else if(parameters->verbosity>=2){
    std::cout<<"Time for learning: "<<(time_all/100)<<"s"<<std::endl;
  };

  return the_result;
};


/**
 *
 * Pattern SVM
 *
 */

int svm_pattern_c::is_alpha_neg(const SVMINT i){
  // variable i is alpha*

  int result;
  if(all_ys[i] > 0){
    result = 1;
  }
  else{
    result = -1;
  };
  return result;
};


SVMFLOAT svm_pattern_c::nabla(const SVMINT i){
  // = is_alpha_neg(i) * sum[i] - 1
  if(all_ys[i]>0){
    return(sum[i]-1);
  }
  else{
    return(-sum[i]-1);
  };
};


SVMFLOAT svm_pattern_c::lambda(const SVMINT i){
  // size lagrangian multiplier of the active constraint

  SVMFLOAT alpha;
  SVMFLOAT result=0;

  alpha=all_alphas[i];

  if(alpha == Cneg){ //-Cneg >= - is_zero){
    // upper bound active
    result = -lambda_eq - sum[i] + 1;
  }
  else if(alpha == 0){ //(alpha <= is_zero) && (alpha >= -is_zero)){
    // lower bound active
    if(all_ys[i]>0){
      result = (sum[i]+lambda_eq) - 1;
    }
    else{
      result = -(sum[i]+lambda_eq) - 1;
    };
  }
  else if(alpha == -Cpos){ //+Cpos <= is_zero){
    // upper bound active
    result = lambda_eq + sum[i] + 1;
  }
  else{
    if(all_ys[i]>0){
      result = -abs(sum[i]+lambda_eq - 1);
    }
    else{
      result = -abs(-sum[i]-lambda_eq - 1);
    };
  };

  return result;
};


int svm_pattern_c::feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)
  int is_feasible=1;

  if(at_bound[i] >= shrink_const){ is_feasible = 0; };

  SVMFLOAT alpha;
  alpha=all_alphas[i];
  if(alpha == Cneg){ //alpha-Cneg >= - is_zero){
    // alpha* at upper bound
    *atbound = 1;
    *the_nabla = sum[i] - 1;
    *the_lambda = -lambda_eq - *the_nabla; //sum[i] + 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if(alpha == 0){
    // lower bound active
    *atbound = -1;
    if(all_ys[i]>0){
      *the_nabla = sum[i] - 1;
      *the_lambda = lambda_eq + *the_nabla; //sum[i]+lambda_eq - 1;
    }
    else{
      *the_nabla = -sum[i] - 1;
      *the_lambda = -lambda_eq + *the_nabla; //-sum[i]-lambda_eq - 1;
    };
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if(alpha == -Cpos){ //alpha+Cpos <= is_zero){
    *atbound = 1;
    *the_nabla = -sum[i] - 1;
    *the_lambda = lambda_eq - *the_nabla; //sum[i] - 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else{
    // not at bound
    *atbound = 0;
    if(all_ys[i]>0){
      *the_nabla = sum[i] - 1;
      *the_lambda = -abs(*the_nabla+lambda_eq);
    }
    else{
      *the_nabla = -sum[i] - 1;
      *the_lambda = -abs(lambda_eq - *the_nabla);
    };
    at_bound[i] = 0;
  };
  if(*the_lambda >= feasible_epsilon){
      is_feasible = 0; 
  };

  return is_feasible;
};


void svm_pattern_c::init_optimizer(){
  // Cs are dived by examples_total in init_optimizer
  svm_c::init_optimizer();
  SVMINT i;
  for(i=0;i<working_set_size;i++){
    qp.l[i] = 0;
  };
};


/**
 *
 * Regression SVM
 *
 */

int svm_regression_c::is_alpha_neg(const SVMINT i){
  // variable i is alpha*
  int result;
  if(all_alphas[i] > 0){
    result = 1;
  }
  else if(all_alphas[i] == 0){
    if(sum[i] - all_ys[i] + lambda_eq>0){
      result = -1;
    }
    else{
      result = 1;
    };
    //    result = 2*((i+at_bound[i])%2)-1;
  }
  else{
    result = -1;
  };
  return result;
};


SVMFLOAT svm_regression_c::nabla(const SVMINT i){
  if(all_alphas[i] > 0){
    return( sum[i] - all_ys[i] + epsilon_neg);
  }
  else if(all_alphas[i] == 0){
    if(is_alpha_neg(i)>0){
      return( sum[i] - all_ys[i] + epsilon_neg);
    }
    else{
      return(-sum[i] + all_ys[i] + epsilon_pos);
    };
  }
  else{
    return(-sum[i] + all_ys[i] + epsilon_pos);
  };
};


SVMFLOAT svm_regression_c::lambda(const SVMINT i){
  // size lagrangian multiplier of the active constraint

  SVMFLOAT alpha;
  SVMFLOAT result = -abs(nabla(i)+is_alpha_neg(i)*lambda_eq);
    //= -infinity; // default = not at bound

  alpha=all_alphas[i];

  if(alpha>is_zero){
    // alpha*
    if(alpha-Cneg >= - is_zero){
      // upper bound active
      result = -lambda_eq-sum[i] + all_ys[i] - epsilon_neg;
    };
  }
  else if(alpha >= -is_zero){
    // lower bound active
    if(all_alphas[i] > 0){
      result = sum[i] - all_ys[i] + epsilon_neg + lambda_eq;
    }
    else if(all_alphas[i] == 0){
      if(is_alpha_neg(i)>0){
	result = sum[i] - all_ys[i] + epsilon_neg + lambda_eq;
      }
      else{
	result = -sum[i] + all_ys[i] + epsilon_pos-lambda_eq;
      };
    }
    else{
      result = -sum[i] + all_ys[i] + epsilon_pos-lambda_eq;
    };
  }
  else if(alpha+Cpos <= is_zero){
    // upper bound active
    result = lambda_eq + sum[i] - all_ys[i] - epsilon_pos;
  };

  return result;
};


int svm_regression_c::feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)
  int is_feasible = 1; // <=> at_bound < shrink and lambda < -feas_eps

  if(at_bound[i] >= shrink_const){ is_feasible = 0; };

  SVMFLOAT alpha;

  alpha=all_alphas[i];

  if(alpha-Cneg >= -is_zero){
    // alpha* at upper bound
    *atbound = 1;
    *the_nabla = sum[i] - all_ys[i] + epsilon_neg;
    *the_lambda = -lambda_eq- *the_nabla;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if((alpha<=is_zero) && (alpha >= -is_zero)){
    // lower bound active
    *atbound = 1;
    if(all_alphas[i] > 0){
      *the_nabla = sum[i] - all_ys[i] + epsilon_neg;
      *the_lambda = *the_nabla + lambda_eq;
    }
    else if(all_alphas[i]==0){
      if(is_alpha_neg(i)>0){
	*the_nabla = sum[i] - all_ys[i] + epsilon_neg;
	*the_lambda = *the_nabla + lambda_eq;
      }
      else{
	*the_nabla = -sum[i] + all_ys[i] + epsilon_pos;
	*the_lambda = *the_nabla - lambda_eq;
      };
    }
    else{
      *the_nabla = -sum[i] + all_ys[i] + epsilon_pos;
      *the_lambda = *the_nabla - lambda_eq;
    };
    if(*the_lambda >= 0){
      if(all_alphas[i] != 0){
	// check both constraints!
	all_alphas[i]=0;
	if(is_alpha_neg(i)>0){
	  *the_lambda = *the_nabla + lambda_eq;
	}
	else{
	  *the_lambda = *the_nabla - lambda_eq;
	};
	if(*the_lambda >= 0){
	  at_bound[i]++;
	};
      }
      else{
	at_bound[i]++;
      };
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if(alpha+Cpos <= is_zero){
    // alpha at upper bound
    *atbound = 1;
    *the_nabla = -sum[i] + all_ys[i] + epsilon_pos;
    *the_lambda = lambda_eq - *the_nabla;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else{
    // not at bound
    *atbound = 0;
    if(is_alpha_neg(i)>0){
      *the_nabla = sum[i] - all_ys[i] + epsilon_neg;
      *the_lambda = -abs(*the_nabla + lambda_eq);
    }
    else{
      *the_nabla = -sum[i] + all_ys[i] + epsilon_pos;
      *the_lambda = -abs(lambda_eq - *the_nabla);
    };
  };
  if(*the_lambda>=feasible_epsilon){
    is_feasible = 0; 
  };
  return is_feasible;
};


