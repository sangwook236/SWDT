#include "svm_nu.h"

/**
 *
 * nu SVM
 *
 */

void svm_nu_regression_c::init(kernel_c* new_kernel, parameters_c* new_parameters){
  svm_c::init(new_kernel,new_parameters);
  lambda_nu = 0;
  sum_alpha_nu = 0;
  qp.m = 2;
  epsilon_pos=0;
  epsilon_neg=0;
  if(new_parameters->realC <= 0){
    new_parameters->realC =1;
  };
  nu = new_parameters->nu * new_parameters->realC;
};


SVMFLOAT svm_nu_regression_c::nabla(const SVMINT i){
  if(is_alpha_neg(i) > 0){
    return( sum[i] - all_ys[i]);
  }
  else{
    return(-sum[i] + all_ys[i]);
  };
};


void svm_nu_regression_c::reset_shrinked(){
  svm_c::reset_shrinked();
  sum_alpha_nu = 0;
};



int svm_nu_regression_c::is_alpha_neg(const SVMINT i){
  // variable i is alpha*
  int result;
  if(all_alphas[i] > 0){
    result = 1;
  }
  else if(all_alphas[i] == 0){
    SVMFLOAT k = sum[i] - all_ys[i] + lambda_eq;
    if(k+lambda_nu>0){
      result = -1;
    }
    else if(-k+lambda_nu>0){
      result = 1;
    }
    else{
      result = 2*((i+at_bound[i])%2)-1;
    };
  }
  else{
    result = -1;
  };
  return result;
};


SVMFLOAT svm_nu_regression_c::lambda(const SVMINT i){
  // size lagrangian multiplier of the active constraint

  SVMFLOAT alpha;
  SVMFLOAT result = 0;

  alpha=all_alphas[i];

  if(alpha>is_zero){
    // alpha*
    if(alpha-Cneg >= - is_zero){
      // upper bound active
      result = -lambda_eq-lambda_nu-nabla(i);
    }
    else{
      result = -abs(nabla(i)+lambda_eq+lambda_nu);
    };
  }
  else if(alpha >= -is_zero){
    // lower bound active
    if(is_alpha_neg(i) > 0){
      result = nabla(i) + lambda_eq+lambda_nu;
    }
    else{
      result = nabla(i)-lambda_eq+lambda_nu;
    };
  }
  else if(alpha+Cpos <= is_zero){
    // upper bound active
    result = lambda_eq -lambda_nu - nabla(i);
  }
  else{
    result = -abs(nabla(i)-lambda_eq+lambda_nu);
  };

  return result;
};


int svm_nu_regression_c::feasible(const SVMINT i){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)

  if(at_bound[i] >= shrink_const){ return 0; };

  SVMFLOAT alpha;
  SVMFLOAT result;

  alpha=all_alphas[i];
  //  feasible_epsilon=-1;
  if(alpha-Cneg >= - is_zero){
    // alpha* at upper bound
    result = -lambda_eq-lambda_nu-nabla(i);
    if(result>=-feasible_epsilon){
      return 0; 
    };
  }
  else if((alpha<=is_zero) && (alpha >= -is_zero)){
    // lower bound active
    if(is_alpha_neg(i) > 0){
      result = nabla(i) + lambda_eq+lambda_nu;
    }
    else{
      result = nabla(i)-lambda_eq+lambda_nu;
    };
    if(result>=-feasible_epsilon){
      return 0; 
    };
  }
  else if(alpha+Cpos <= is_zero){
    // alpha at upper bound
    result = lambda_eq -lambda_nu- nabla(i);
    if(result>=-feasible_epsilon){
      return 0; 
    };
  }
  else{
    // not at bound
    result= abs(nabla(i)+is_alpha_neg(i)*lambda_eq+lambda_nu);
    if(result<=feasible_epsilon){
      return 0; 
    };
  };
  return 1;
};


void svm_nu_regression_c::init_optimizer(){
  lambda_nu_WS = 0;
  lambda_nu = 0;
  sum_alpha_nu=0;
  qp.m = 2;
  svm_c::init_optimizer();
  smo.init(parameters->is_zero,parameters->convergence_epsilon,working_set_size*working_set_size*10);
  epsilon_pos=0;
  epsilon_neg=0;
  Cpos /= (SVMFLOAT)examples_total;
  Cneg /= (SVMFLOAT)examples_total;
};


void svm_nu_regression_c::project_to_constraint(){
  SVMFLOAT alpha;
  SVMFLOAT my_alpha_sum=sum_alpha;
  SVMFLOAT my_alpha_nu_sum=sum_alpha_nu-nu;
  SVMINT total_pos = 0;
  SVMINT total_neg = 0;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    alpha = all_alphas[i];
    my_alpha_sum += alpha;
    my_alpha_nu_sum += abs(alpha);
    if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
      total_neg++;
    }
    else if((alpha<-is_zero) && (alpha+Cpos > is_zero)){
      total_pos++;
    };
  };

  //project
  if((total_pos > 0) && (total_neg > 0)){
    for(i=0;i<examples_total;i++){
      alpha = all_alphas[i];
      if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
	all_alphas[i] -= (my_alpha_sum+my_alpha_nu_sum)/2/total_neg;
      }
      else if((alpha<-is_zero) && (alpha+Cpos > is_zero)){
	all_alphas[i] -= (my_alpha_sum-my_alpha_nu_sum)/2/total_pos;
      };
    };
  };
};


int svm_nu_regression_c::convergence(){
  long time_start = get_time();
  SVMFLOAT pos_sum = 0;
  SVMFLOAT neg_sum = 0;
  SVMINT total=0;
  SVMFLOAT alpha_sum=0;
  SVMFLOAT alpha_nu_sum=0;
  SVMFLOAT alpha=0;
  SVMINT i;
  int result=1;

  // actual convergence-test
  total = 0; alpha_sum=0;

  SVMINT total_pos = 0;
  SVMINT total_neg = 0;

  for(i=0;i<examples_total;i++){
    alpha = all_alphas[i];
    alpha_sum += alpha;
    alpha_nu_sum += abs(alpha);
    if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
      // alpha^* => nabla = lambda_eq + lambda_nu
      neg_sum += nabla(i); //sum[i];
      total_neg++;
    }
    else if((alpha<-is_zero) && (alpha+Cpos > is_zero)){
      // alpha => nabla = -lambda_eq + lambda_nu
      pos_sum += nabla(i); //-sum[i];
      total_pos++;
    };
  };

  if((total_pos>0) && (total_neg > 0)){
    lambda_nu = -(neg_sum/total_neg+pos_sum/total_pos)/2;
    lambda_eq = -(neg_sum/total_neg-pos_sum/total_pos)/2;
    if(target_count>2){
      if(parameters->verbosity>=5){
	std::cout<<"Re-estimating lambdas from WS"<<std::endl;
      };
      total_pos=0; total_neg=0;
      pos_sum = 0; neg_sum = 0;
      // estimate lambdas from WS
      for(i=0;i<working_set_size;i++){
	alpha = all_alphas[working_set[i]];
	if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
	  // alpha^* => nabla = lambda_eq + lambda_nu
	  neg_sum += nabla(working_set[i]);
	  total_neg++;	  
	}
	else if((alpha<-is_zero) && (alpha+Cpos > is_zero)){
	  // alpha => nabla = -lambda_eq + lambda_nu
	  pos_sum += nabla(working_set[i]);
	  total_pos++;
	};
      };
      if((total_pos>0) && (total_neg > 0)){
	lambda_nu = -(neg_sum/total_neg+pos_sum/total_pos)/2;
	lambda_eq = -(neg_sum/total_neg-pos_sum/total_pos)/2;
      };
      if(target_count>30){
	i = working_set[target_count%working_set_size];
	if(parameters->verbosity>=5){
	  std::cout<<"Setting lambdas to nabla("<<i<<")"<<std::endl;
	};
	if(is_alpha_neg(i) > 0){
	  lambda_eq = -nabla(i);
	}
	else{
	  lambda_eq = nabla(i);
	  };
	lambda_nu=0;
      };
    };

    if(parameters->verbosity>= 4){
      std::cout<<"lambda_eq = "<<lambda_eq<<std::endl;
      std::cout<<"lambda_nu = "<<lambda_nu<<std::endl;
    };
  }
  else{
    // lambda_eq and lambda_nu are a bit harder to find:
    SVMFLOAT max1=-infinity;
    SVMFLOAT max2=-infinity;
    SVMFLOAT max3=-infinity;
    SVMFLOAT max4=-infinity;
    SVMFLOAT the_nabla;
    for(i=0;i<examples_total;i++){
      alpha = all_alphas[i];
      the_nabla = nabla(i);
      if(alpha-Cneg>=-is_zero){
	if(the_nabla>max4) max4 = the_nabla;
      }
      else if(alpha+Cpos<=is_zero){
	if(the_nabla>max3) max3 = the_nabla;
      }
      else if((alpha <= is_zero) && (alpha>=-is_zero)){
	if(is_alpha_neg(i)){
	  if(the_nabla>max1) max1 = the_nabla;
	}
	else{
	  if(the_nabla>max2) max2 = the_nabla;
	};
      };
    };

    //    std::cout<<"max = ("<<max1<<", "<<max2<<", "<<max3<<", "<<max4<<std::endl;
    
    if(max1==-infinity) max1=0;
    if(max2==-infinity) max2=0;
    if(max3==-infinity) max3=0;
    if(max4==-infinity) max4=0;
    lambda_eq = (max1+max3-max2-max4)/4;
    lambda_nu = (max1+max2-max3-max4)/4;

    //    std::cout<<((max1+max3)/2)<<" <= lambda_eq <= "<<(-(max2+max4)/2)<<std::endl;
    //    std::cout<<((max1+max2)/2)<<" <= lambda_nu <= "<<(-(max3+max4)/2)<<std::endl;

    if(parameters->verbosity>= 4){
      std::cout<<"*** no SVs in convergence(), lambda_eq = "<<lambda_eq<<"."<<std::endl;
      std::cout<<"                             lambda_nu = "<<lambda_nu<<"."<<std::endl;
    };
  };

  // check linear constraint
  if(abs(alpha_sum+sum_alpha) > convergence_epsilon){
    // equality constraint violated
    if(parameters->verbosity>= 3){
      std::cout<<"alpha_sum "<<alpha_sum<<std::endl;
      std::cout<<"sum_alpha "<<sum_alpha<<std::endl;
      std::cout<<"No convergence: equality constraint violated: |"<<(alpha_sum+sum_alpha)<<"| >> 0"<<std::endl;
    };
    project_to_constraint();
    result = 0;  
  };
  // note: original nu is already multiplied by C in init_optimizer
  if(abs(alpha_nu_sum+sum_alpha_nu-nu) > convergence_epsilon){
    // equality constraint violated
    if(parameters->verbosity>= 3){
      std::cout<<"alpha_nu_sum "<<alpha_nu_sum<<std::endl;
      std::cout<<"sum_alpha_nu "<<sum_alpha_nu<<std::endl;
      std::cout<<"No convergence: nu-equality constraint violated: |"<<(alpha_nu_sum+sum_alpha_nu-nu)<<"| >> 0"<<std::endl;
    };
    project_to_constraint();
    result = 0;  
  };

  i=0;
  while((i<examples_total) && (result != 0)){
    if(lambda(i)>=-convergence_epsilon){
      i++;
    }
    else{
      result = 0;
    };
  };

  time_convergence += get_time() - time_start;
  return result;
};


void svm_nu_regression_c::init_working_set(){
  if((((SVMFLOAT)examples_total) * nu) > parameters->realC*(((SVMFLOAT)examples_total) - ((SVMFLOAT)(examples_total%2)))){
    nu = (((SVMFLOAT)examples_total) - ((SVMFLOAT)(examples_total%2))) /
      ((SVMFLOAT)examples_total);
    nu *= parameters->realC;
    nu -= is_zero; // just to make sure
    std::cout<<"ERROR: nu too large, setting nu = "<<nu<<std::endl;
  };

  if(examples->initialised_alpha()){
    // check bounds
    project_to_constraint();
  };

  // calculate nu-sum
  sum_alpha_nu=0;
  SVMFLOAT the_nu_sum = 0;
  SVMFLOAT the_sum=0;
  SVMINT ni;
  for(ni=0;ni<examples_total;ni++){
    the_sum += all_alphas[ni];
    the_nu_sum += abs(all_alphas[ni]);
  };

  if((abs(the_sum) > is_zero) || (abs(the_nu_sum-nu) > is_zero)){
    // set initial feasible point
    // neg alpha: -nu/2n
    // pos alpha:  nu/2p

    SVMFLOAT new_nu_alpha = nu/((SVMFLOAT)(examples_total-examples_total%1));

    for(ni=0;ni<examples_total/2;ni++){
      examples->put_alpha(ni,new_nu_alpha);
      examples->put_alpha(examples_total-1-ni,-new_nu_alpha);
    };
    if(examples_total%2 != 0){
      examples->put_alpha(1+examples_total/2,0);
    };
    examples->set_initialised_alpha();
  };

  svm_c::init_working_set();   
};


void svm_nu_regression_c::shrink(){
  // move shrinked examples to back
  if(to_shrink>examples_total/20){
    SVMINT i;
    SVMINT last_pos=examples_total;
    for(i=0;i<last_pos;i++){
      if(at_bound[i] >= shrink_const){
	// shrink i
	sum_alpha += all_alphas[i];
	sum_alpha_nu += is_alpha_neg(i)*all_alphas[i];
	last_pos--;
	examples->swap(i,last_pos);
	kernel->overwrite(i,last_pos);
	sum[i] = sum[last_pos];
	at_bound[i] = at_bound[last_pos];
      };
    };
    
    examples_total = last_pos;
    kernel->set_examples_size(examples_total);
    
    if(parameters->verbosity>=4){
      std::cout<<"shrinked to "<<examples_total<<" variables"<<std::endl;
    };
  };
};


void svm_nu_regression_c::optimize(){
  // optimizer-specific call
  // get time
  long time_start = get_time();

  qp.n = working_set_size;

  SVMINT i;
  SVMINT j;

  // equality constraint
  qp.b[0] = 0;
  qp.b[1] = 0;

  for(i=0;i<working_set_size;i++){
    qp.b[0] += all_alphas[working_set[i]];
    if(qp.A[i] > 0){
      qp.b[1] += all_alphas[working_set[i]];
    }
    else{
      qp.b[1] -= all_alphas[working_set[i]];
    };
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
  SVMFLOAT new_nu_constraint_sum=0;
  SVMINT pos_count = 0;
  SVMINT neg_count = 0;

  // optimize
  int KKTerror=1;
  int convError=0;

  if(feasible_epsilon>is_zero){
    smo.set_max_allowed_error(feasible_epsilon);
  };

  int result = smo.smo_solve_const_sum(&qp,primal);
  lambda_WS = smo.get_lambda_eq();
  lambda_nu_WS = smo.get_lambda_nu();

  if(parameters->verbosity>=5){
    std::cout<<"smo ended with result "<<result<<std::endl;
    std::cout<<"lambda_WS = "<<lambda_WS<<std::endl;
    std::cout<<"lambda_nu_WS = "<<lambda_nu_WS<<std::endl;
  };

  // clip
  new_constraint_sum=qp.b[0];
  new_nu_constraint_sum=qp.b[1];
  for(i=0;i<working_set_size;i++){
    // check if at bound
    if(primal[i] <= is_zero){
      // at lower bound
      primal[i] = qp.l[i];
    }
    else if(primal[i] - qp.u[i] >= -is_zero){
      // at upper bound
      primal[i] = qp.u[i];
    };                            
    new_constraint_sum -= qp.A[i]*primal[i];
    new_nu_constraint_sum -= primal[i];
  };

  // enforce equality constraint
  pos_count=0;
  neg_count=0;
  for(i=0;i<working_set_size;i++){
    if((primal[i] < qp.u[i]) && (primal[i] > qp.l[i])){
      if(qp.A[i]>0){
	pos_count++;
      }
      else{
	neg_count++;
      };
    };
  };
  if((pos_count>0) && (neg_count>0)){
    SVMFLOAT pos_add = (new_constraint_sum+new_nu_constraint_sum)/(2*pos_count);
    SVMFLOAT neg_add = (new_constraint_sum-new_nu_constraint_sum)/(2*neg_count);
    //    std::cout<<"Adding ("<<pos_add<<", "<<neg_add<<")"<<std::endl;
    for(i=0;i<working_set_size;i++){
      if((primal[i] > qp.l[i]) && 
	 (primal[i] < qp.u[i])){
	// real sv
	if(qp.A[i]>0){
	  primal[i] += pos_add;
	}
	else{
	  primal[i] -= neg_add;
	};
      };
    };
  }
  else if((abs(new_constraint_sum)   > is_zero) ||
	  (abs(new_nu_constraint_sum) > is_zero)){
    // error, can't get feasible point
    if(parameters->verbosity >= 5){
      std::cout<<"WARNING: No SVs, constraint_sum = "<<new_constraint_sum<<std::endl
	  <<"              nu-constraint_sum = "<<new_nu_constraint_sum<<std::endl;
    };
    old_target = -1e20; 
    convError=1;
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
    //      target_count=0;
    if(parameters->descend < old_target - new_target){
      target_count=0;
    }
    else{
      convError=1;
    };
    if(parameters->verbosity>=5){
      std::cout<<"descend = "<<old_target-new_target<<"  ("<<old_target<<" --> "<<new_target<<")"<<std::endl;
    };
  }
  else{
    // nothing we can do
    KKTerror=0;
    convError=1;
    if(parameters->verbosity>=5){
      std::cout<<"WARNING: no descend ("<<old_target<<" -> "<<new_target<<"), stopping"<<std::endl;
    };
  };

  if(convError){
    target_count++;
    if(new_target>=old_target){
      for(i=0;i<working_set_size;i++){
	primal[i] = qp.A[i]*all_alphas[working_set[i]];
      };  
    };                                                                        
    if(parameters->verbosity>=5){	
      std::cout<<"WARNING: Convergence error, setting sigfig = "<<sigfig_max<<std::endl;
    };
  };

  if(target_count>50){
    // non-recoverable numerical error
    feasible_epsilon=1;
    convergence_epsilon*=2;
    if(parameters->verbosity>=1)
      std::cout<<"WARNING: reducing KKT precision to "<<convergence_epsilon<<std::endl;
    target_count=0;
  };

  if(parameters->verbosity>=5){	
    std::cout<<"Resulting values:"<<std::endl;
    for(i=0;i<working_set_size;i++){
      std::cout<<i<<": "<<primal[i]<<std::endl;
    };
  };

  time_optimize += get_time() - time_start;
};


void svm_nu_regression_c::print_special_statistics(){
  // calculate tube size epsilon
  SVMFLOAT b = examples->get_b();
  SVMFLOAT epsilon_pos = 0;
  SVMFLOAT epsilon_neg = 0;
  SVMINT pos_count = 0;
  SVMINT neg_count = 0;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    if((all_alphas[i] > is_zero) && (all_alphas[i]-Cpos<-is_zero)){
      epsilon_neg += all_ys[i]-sum[i]-b;
      neg_count++;
    }
    else if((all_alphas[i] <- is_zero) && (all_alphas[i]+Cneg>+is_zero)){
      epsilon_pos += -all_ys[i]+sum[i]+b;
      pos_count++;
    };
  };
  if((parameters->Lpos == parameters->Lneg) ||
     (pos_count == 0) ||
     (neg_count == 0)){
    // symmetrical
    epsilon_pos += epsilon_neg;
    pos_count += neg_count;
    if(pos_count>0){
      epsilon_pos /= (SVMINT)pos_count;
      std::cout<<"epsilon = "<<epsilon_pos<<std::endl;
    }
    else{
      std::cout<<"ERROR: could not calculate epsilon."<<std::endl;
      std::cout<<pos_count<<"\t"<<neg_count<<std::endl; // @@@@@@@
    };
  }
  else{
    // asymmetrical
    epsilon_pos /= (SVMINT)pos_count;
    std::cout<<"epsilon+ = "<<epsilon_pos<<std::endl;
    epsilon_neg /= (SVMINT)neg_count;
    std::cout<<"epsilon- = "<<epsilon_pos<<std::endl;
  };
};


/**
 *
 * svm_nu_pattern_c
 *
 **/

SVMFLOAT svm_nu_pattern_c::nabla(const SVMINT i){
  if(all_ys[i] > 0){
    return( sum[i]);
  }
  else{
    return(-sum[i]);
  };
};


void svm_nu_pattern_c::init(kernel_c* new_kernel, parameters_c* new_parameters){
  new_parameters->realC = 1;
  svm_nu_regression_c::init(new_kernel,new_parameters);
};


void svm_nu_pattern_c::init_optimizer(){
  // Cs are dived by examples_total in init_optimizer
  svm_nu_regression_c::init_optimizer();
  SVMINT i;
  for(i=0;i<working_set_size;i++){
    qp.l[i] = 0;
  };
};


void svm_nu_pattern_c::update_working_set(){
  svm_c::update_working_set();
  SVMINT i;
  for(i=0;i<working_set_size;i++){
    if(qp.A[i]>0){
      qp.c[i] += all_ys[working_set[i]];
    }
    else{
      qp.c[i] -= all_ys[working_set[i]];
    };
  };
};


void svm_nu_pattern_c::init_working_set(){
  // calculate nu-sum 

  if(examples->initialised_alpha()){
    project_to_constraint();
  };

  sum_alpha_nu=0;
  SVMFLOAT the_nu_sum = 0;
  SVMFLOAT the_sum=0;
  SVMINT pos_count=0;
  SVMINT neg_count=0;
  SVMINT ni;
  for(ni=0;ni<examples_total;ni++){
    the_sum += all_alphas[ni];
    the_nu_sum += abs(all_alphas[ni]);
    if(is_alpha_neg(ni)> 0){
      neg_count++;
    }
    else{
      pos_count++;
    };
  };

  if((abs(the_sum) > is_zero) || (abs(the_nu_sum-nu) > is_zero)){
    // set initial feasible point
    // neg alpha: -nu/2n
    // pos alpha:  nu/2p

    if((nu*(SVMFLOAT)examples_total>2*(SVMFLOAT)pos_count) ||
       (nu*(SVMFLOAT)examples_total>2*(SVMFLOAT)neg_count)){
      nu = 2*((SVMFLOAT)pos_count)/((SVMFLOAT)examples_total);
      if(nu > 2*((SVMFLOAT)neg_count)/((SVMFLOAT)examples_total)){
	nu = 2*((SVMFLOAT)neg_count)/((SVMFLOAT)examples_total);
      };
      nu -= is_zero; // just to make sure
      std::cout<<"ERROR: nu too large, setting nu = "<<nu<<std::endl;
    };

    for(ni=0;ni<examples_total;ni++){
      if(is_alpha_neg(ni)> 0){
	examples->put_alpha(ni,nu/(2*(SVMFLOAT)neg_count));
      }
      else{
	examples->put_alpha(ni,-nu/(2*(SVMFLOAT)pos_count));
      };
    };
    examples->set_initialised_alpha();
  };

  svm_c::init_working_set();
};


void svm_nu_pattern_c::print_special_statistics(){
  // calculate margin rho
  SVMFLOAT b = examples->get_b();
  SVMFLOAT rho_pos = 0;
  SVMFLOAT rho_neg = 0;
  SVMINT pos_count = 0;
  SVMINT neg_count = 0;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    if((all_alphas[i] > is_zero) && (all_alphas[i]-Cpos<-is_zero)){
      rho_neg += sum[i]+b;
      neg_count++;
    }
    else if((all_alphas[i] <- is_zero) && (all_alphas[i]+Cneg>+is_zero)){
      rho_pos += -sum[i]-b;
      pos_count++;
    };
  };
  if((parameters->Lpos == parameters->Lneg) ||
     (pos_count == 0) ||
     (neg_count == 0)){
    // symmetrical
    rho_pos += rho_neg;
    pos_count += neg_count;
    if(pos_count>0){
      rho_pos /= (SVMINT)pos_count;
      std::cout<<"margin = "<<rho_pos<<std::endl;
    }
    else{
      std::cout<<"ERROR: could not calculate margin."<<std::endl;
    };
  }
  else{
    // asymmetrical
    rho_pos /= (SVMINT)pos_count;    std::cout<<"margin+ = "<<rho_pos<<std::endl;
    rho_neg /= (SVMINT)neg_count;
    std::cout<<"margin- = "<<rho_pos<<std::endl;
  };
};


/**
 *
 * svm_distribution_c
 *
 **/

int svm_distribution_c::is_alpha_neg(const SVMINT i){
  // variable i is alpha*
  return 1;
};


SVMFLOAT svm_distribution_c::nabla(const SVMINT i){
  return( sum[i]);
};


SVMFLOAT svm_distribution_c::lambda(const SVMINT i){
  // size lagrangian multiplier of the active constraint

  SVMFLOAT alpha;
  SVMFLOAT result = 0;

  alpha=all_alphas[i];

  if(alpha>is_zero){
    // alpha*
    if(alpha-Cneg >= - is_zero){
      // upper bound active
      result = -lambda_eq-sum[i];
    }
    else{
      result = -abs(sum[i]+lambda_eq);
    };
  }
  else{
    // lower bound active
    result = sum[i] + lambda_eq;
  };

  return result;
};


int svm_distribution_c::feasible(const SVMINT i){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)

  if(at_bound[i] >= shrink_const){ return 0; };

  SVMFLOAT alpha;
  SVMFLOAT result;

  alpha=all_alphas[i];

  if(alpha-Cneg >= - is_zero){
    // alpha* at upper bound
    result = -lambda_eq - sum[i];
    if(result>=-feasible_epsilon){
      return 0; 
    };
  }
  else if(alpha<=is_zero){
    // lower bound active
    result = sum[i]+lambda_eq;
    if(result>=-feasible_epsilon){
      return 0; 
    };
  }
  else{
    // not at bound
    result= abs(sum[i]+lambda_eq);
    if(result<=feasible_epsilon){
      return 0; 
    };
  };
  return 1;
};


int svm_distribution_c::feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound){
  // is direction i feasible to minimize the target function
  // (includes which_alpha==0)
  int is_feasible=1;

  if(at_bound[i] >= shrink_const){ is_feasible = 0; };

  SVMFLOAT alpha;

  alpha=all_alphas[i];
  *the_nabla = sum[i];

  if(alpha >= Cneg){ //alpha-Cneg >= - is_zero){
    // alpha* at upper bound
    *atbound = 1;
    *the_lambda = -lambda_eq - *the_nabla; //sum[i] + 1;
    if(*the_lambda >= 0){
      at_bound[i]++;
      if(at_bound[i] == shrink_const) to_shrink++;
    }
    else{
      at_bound[i] = 0;
    };
  }
  else if(alpha <= 0){
    // lower bound active
    *atbound = -1;
    *the_lambda = lambda_eq + *the_nabla; //sum[i] + 1;
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
    *the_lambda = -abs(*the_nabla+lambda_eq);
    at_bound[i] = 0;
  };
  if(*the_lambda >= feasible_epsilon){
    is_feasible = 0; 
  };

  return is_feasible;
};


void svm_distribution_c::init(kernel_c* new_kernel, parameters_c* new_parameters){
  new_parameters->realC = 1;
  nu = new_parameters->nu;
  convergence_epsilon = 1e-4;
  svm_pattern_c::init(new_kernel,new_parameters);
  //  is_pattern = 1;
};

void svm_distribution_c::init_optimizer(){
  // Cs are dived by examples_total in init_optimizer
  svm_pattern_c::init_optimizer();
};


void svm_distribution_c::project_to_constraint(){
  SVMINT total = 0;
  SVMFLOAT alpha_sum=sum_alpha-nu;
  SVMFLOAT alpha=0;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    alpha = all_alphas[i];
    alpha_sum += alpha;
    if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
      total++;
    };
  };
  if(total>0){
    // equality constraint violated
    alpha_sum /= (SVMFLOAT)total;
    for(i=0;i<examples_total;i++){
      if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
	all_alphas[i] -= alpha_sum;
      };
    };
  };
};


int svm_distribution_c::convergence(){
  long time_start = get_time();
  SVMFLOAT the_lambda_eq = 0;
  SVMINT total = 0;
  SVMFLOAT alpha_sum=0;
  SVMFLOAT alpha=0;
  SVMINT i;
  int result=1;

  // actual convergence-test
  total = 0; alpha_sum=0;
  //  std::cout<<Cneg<<"\t"<<nu<<"\t"<<all_alphas[0]<<std::endl;
  for(i=0;i<examples_total;i++){
    alpha = all_alphas[i];
    alpha_sum += alpha;
    if((alpha>is_zero) && (alpha-Cneg < -is_zero)){
      // alpha^* = - nabla
      the_lambda_eq += -sum[i];
      total++;
    };
  };

  if(parameters->verbosity>= 4){
    std::cout<<"lambda_eq = "<<(the_lambda_eq/total)<<std::endl;
  };
  if(total>0){
    lambda_eq = the_lambda_eq / total;
  }
  else{
    // keep WS lambda_eq
    lambda_eq = lambda_WS;
    if(parameters->verbosity>= 4){
      std::cout<<"*** no SVs in convergence(), lambda_eq = "<<lambda_eq<<"."<<std::endl;
    };
  };

  if(target_count>2){
    if(target_count>20){
      // desperate!
      lambda_eq = ((40-target_count)*lambda_eq + (target_count-20)*lambda_WS)/20;
      if(parameters->verbosity>=5){
	std::cout<<"Re-Re-calculated lambda from WS: "<<lambda_eq<<std::endl;
      };
      if(target_count>40){
	// really desperate, kick one example out!
	i = working_set[target_count%working_set_size];
	lambda_eq = -sum[i];
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
  if(abs(alpha_sum+sum_alpha-nu) > convergence_epsilon){
    // equality constraint violated
    if(parameters->verbosity>= 4){
      std::cout<<"No convergence: equality constraint violated: |"<<(alpha_sum+sum_alpha)<<"| >> 0"<<std::endl;
    };
    project_to_constraint();
    result = 0;  
  };

  i=0;
  while((i<examples_total) && (result != 0)){
    if(lambda(i)>=-convergence_epsilon){
      i++;
    }
    else{
      result = 0;
    };
  };

  time_convergence += get_time() - time_start;
  return result;
};


void svm_distribution_c::init_working_set(){
  // calculate sum
  SVMINT i,j;

  if(nu>1){
    std::cout<<"ERROR: nu too large, setting nu to 1"<<std::endl;
    nu = 1-is_zero;
  };
  SVMFLOAT the_sum=0;
  for(i=0; i<examples_total;i++){
    the_sum += all_alphas[i];
  };
  if(abs(the_sum-nu) > is_zero){
    for(i=0; i<examples_total;i++){
      examples->put_alpha(i,nu/((SVMFLOAT)examples_total));
    };
    examples->set_initialised_alpha();
  };

  if(parameters->verbosity >= 3){
    std::cout<<"Initialising variables, this may take some time."<<std::endl;
  };
  for(i=0; i<examples_total;i++){
    all_ys[i] = 1;
    sum[i] = 0;
    at_bound[i] = 0;
    for(j=0; j<examples_total;j++){
      sum[i] += all_alphas[j]*kernel->calculate_K(i,j);
    };
  };

  calculate_working_set();
  update_working_set();
};


void svm_distribution_c::print_special_statistics(){
  // calculate margin rho
  SVMFLOAT rho = 0;
  SVMINT count = 0;
  SVMFLOAT norm_x;
  SVMFLOAT max_norm_x=-infinity;
  //  SVMFLOAT xi_i;
  //  SVMINT estim_loo=examples_total;
  //  SVMINT estim_loo2=examples_total;
  SVMINT svs=0;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    if((all_alphas[i] > is_zero) && (all_alphas[i]-Cpos<-is_zero)){
      rho += sum[i];
      count++;
    };
    if(all_alphas[i] != 0){
      svs++;
      norm_x = kernel->calculate_K(i,i);
      if(norm_x>max_norm_x){
	max_norm_x = norm_x;
      };
    };
  };
  if(count == 0){
    std::cout<<"ERROR: could not calculate margin."<<std::endl;
  }
  else{
    // put -rho as b (same decision function)
    rho /= (SVMINT)count;
    examples->put_b(-rho);
    std::cout<<"margin = "<<rho<<std::endl;
  };

  std::cout<<"examples in distribution support : "<<count<<" ("<<((SVMINT)(10000.0*(SVMFLOAT)count/((SVMFLOAT)examples_total)))/100.0<<"%)."<<std::endl;


};


