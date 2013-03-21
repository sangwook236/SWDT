#include "parameters.h"

parameters_c::parameters_c(){
  clear();
};


void parameters_c::clear(){
  // machine parameters
  realC=0; // 0 => SVM sets default 1/avg(||x||^2)
  nu=0;

  // search optimal C?
  search_c = 'n';
  c_min = 1;
  c_max = c_min;
  c_delta = 1;
  search_stop=0;

  // loss function
  Lpos=1; Lneg=1;
  balance_cost=0;
  quadraticLossPos=0; quadraticLossNeg=0;
  epsilon_pos=0; epsilon_neg=0;
  is_pattern = 0;
  is_linear = 1;
  is_distribution = 0;
  is_nu=0;
  biased = 1;

  // cross-validation?
  cross_validation = 0;
  cv_window = 0;
  cv_inorder = 0;

  // scaling
  do_scale = 1;
  do_scale_y = 1;

  // numerical optimization parameters
  is_zero=1e-10;
  descend = 1e-15;
  max_iterations=100000;
  working_set_size=10;
  convergence_epsilon=1e-3;
  shrink_const = 50;
  kernel_cache=256;

  use_min_prediction = 0;

  // verbosity
  verbosity=3;
  print_w = 0;
  loo_estim = 0;

  // example formats
  default_example_format.sparse = 0;
  default_example_format.where_x = 2;
  default_example_format.where_y = 1;
  default_example_format.where_alpha = 0;
  default_example_format.delimiter = ' ';
};


std::istream& operator >> (std::istream& data_stream, parameters_c& the_parameters){
  //  string s;
  char* s = new char[MAXCHAR];
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  while((! data_stream.eof()) && (next != '@')){
    if(('\n' == next) ||
       (' ' == next) ||
       ('\r' == next) ||
       ('\f' == next) ||
       ('\t' == next)){    
      next = data_stream.get();
    }
    else if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
      //   getline(data_stream,s);
    }
    else{
      // trying to read parameter
      data_stream >> s;
      if((0 == strcmp("C",s)) || (0 == strcmp("c",s))){
	data_stream >> the_parameters.realC;
      }
      else if(0 == strcmp("L",s)){
	data_stream >> the_parameters.Lpos;
	if(the_parameters.Lpos <= 0){
	  throw read_exception("Invalid value for parameter 'L'");
	};
	the_parameters.Lneg = the_parameters.Lpos;
      }
      else if((0 == strcmp("L+",s)) || (0 == strcmp("Lpos",s))){
	data_stream >> the_parameters.Lpos;
	if(the_parameters.Lpos < 0){
	  throw read_exception("Invalid value for parameter 'L+'");
	};
      }
      else if((0 == strcmp("L-",s)) || (0 == strcmp("Lneg",s))){
	data_stream >> the_parameters.Lneg;
	if(the_parameters.Lneg < 0){
	  throw read_exception("Invalid value for parameter 'L-'");
	};
      }
      else if(0 == strcmp("epsilon",s)){
	data_stream >> the_parameters.epsilon_pos;
	if(the_parameters.epsilon_pos < 0){
	  throw read_exception("Invalid value for parameter 'epsilon'");
	};
	the_parameters.epsilon_neg = the_parameters.epsilon_pos;
      }
      else if(0 == strcmp("epsilon+",s)){
	data_stream >> the_parameters.epsilon_pos;
	if(the_parameters.epsilon_pos < 0){
	  throw read_exception("Invalid value for parameter 'epsilon+'");
	};
      }
      else if(0 == strcmp("epsilon-",s)){
	data_stream >> the_parameters.epsilon_neg;
	if(the_parameters.epsilon_neg < 0){
	  throw read_exception("Invalid value for parameter 'epsilon-'");
	};
      }
      else if(0 == strcmp("distribution",s)){
	the_parameters.is_distribution = 1;
	the_parameters.is_pattern = 1;
      }
      else if(0 == strcmp("biased",s)){
	the_parameters.biased = 1;
      }
      else if(0 == strcmp("unbiased",s)){
	the_parameters.biased = 0;
      }
      else if(0 == strcmp("balance_cost",s)){
	the_parameters.balance_cost = 1;
      }
      else if(0 == strcmp("nu",s)){
	data_stream >> the_parameters.nu;
	the_parameters.is_nu = 1;
	if(the_parameters.nu < 0){
	  throw read_exception("Invalid value for parameter 'nu'");
	};
      }
      else if((0 == strcmp("quadraticLoss+",s))|| (0 == strcmp("quadraticloss+",s))){
	the_parameters.quadraticLossPos = 1;
      }
      else if((0 == strcmp("quadraticLoss-",s))||(0 == strcmp("quadraticloss-",s))){
	the_parameters.quadraticLossNeg = 1;
      }
      else if((0 == strcmp("quadraticLoss",s))||(0 == strcmp("quadraticloss",s))){
	the_parameters.quadraticLossPos = 1;
	the_parameters.quadraticLossNeg = 1;
      }
      else if((0 == strcmp("search_C",s)) || (0 == strcmp("search_c",s))){
	data_stream >> s;
	char search_c = s[0]; //(s.c_str())[0];
	if((search_c == 'N') ||(search_c == 'n')){
	  the_parameters.search_c = 'n';
	}
	else if((search_c == 'A') ||(search_c == 'a')){
	  the_parameters.search_c = 'a';
	}
	else if((search_c == 'M') ||(search_c == 'm')){
	  the_parameters.search_c = 'm';
	}
	else if((search_c == 'G') ||(search_c == 'g')){
	  the_parameters.search_c = 'g';
	}
	else{
	  throw read_exception("Invalid value for parameter 'search_C'");
	};
      }
      else if(0 == strcmp("search_stop",s)){
	data_stream >> the_parameters.search_stop;
	if(the_parameters.search_stop < 0){ 
	  throw read_exception("Invalid value for parameter 'search_stop'");
	};
      }
      else if((0 == strcmp("Cmin",s)) || (0 == strcmp("cmin",s))){
	data_stream >> the_parameters.c_min;
	if(the_parameters.c_min < 0){ 
	  throw read_exception("Invalid value for parameter 'Cmin'");
	};
      }
      else if((0 == strcmp("Cmax",s)) || (0 == strcmp("cmax",s))){
	data_stream >> the_parameters.c_max;
	if(the_parameters.c_max <= 0){ 
	  throw read_exception("Invalid value for parameter 'Cmax'");
	};
      }
      else if((0 == strcmp("Cdelta",s)) || (0 == strcmp("cdelta",s))){
	data_stream >> the_parameters.c_delta;
	if(the_parameters.c_delta <= 0){ 
	  throw read_exception("Invalid value for parameter 'Cdelta'");
	};
      }
      else if(0 == strcmp("pattern",s)){
	the_parameters.is_pattern = 1;
      }
      else if(0 == strcmp("regression",s)){
	the_parameters.is_pattern = 0;
      }
      else if(0 == strcmp("scale",s)){
	the_parameters.do_scale = 1;
	the_parameters.do_scale_y = 1;
      }
      else if(0 == strcmp("no_scale",s)){
	the_parameters.do_scale = 0;
	the_parameters.do_scale_y = 0;
      }
      else if((0 == strcmp("cross_validation",s)) || (0 == strcmp("cv",s))){
	if(data_stream.peek() == '\n'){
	  the_parameters.cross_validation = 10;
	}
	else{
	  data_stream >> the_parameters.cross_validation;	
	  if(the_parameters.cross_validation < 0){
	    throw read_exception("Invalid value for parameter 'cross_validation'");
	  };
	};
      }
      else if(0 == strcmp("cv_window",s)){
	data_stream >> the_parameters.cv_window;
	if(the_parameters.cv_window<0){
	  throw read_exception("Invalid value for parameter 'cv_window'");
	};
	the_parameters.cv_inorder = 1;
      }
      else if(0 == strcmp("cv_inorder",s)){
	the_parameters.cv_inorder = 1;
      }
      else if(0 == strcmp("working_set_size",s)){
	data_stream >> the_parameters.working_set_size;
	if(the_parameters.working_set_size < 2){
	  throw read_exception("Invalid value for parameter 'working_set_size'");
	};
      }
      else if(0 == strcmp("max_iterations",s)){
	data_stream >> the_parameters.max_iterations;
	if(the_parameters.max_iterations<0){
	  throw read_exception("Invalid value for parameter 'max_iterations'");
	};
      }
      else if(0 == strcmp("shrink_const",s)){
	data_stream >> the_parameters.shrink_const;
	if(the_parameters.shrink_const<1){
	  throw read_exception("Invalid value for parameter 'shrink_const'");
	};
      }
      else if(0 == strcmp("descend",s)){
	data_stream >> the_parameters.descend;
	if(the_parameters.shrink_const<0){
	  throw read_exception("Invalid value for parameter 'descend'");
	};
      }
      else if(0 == strcmp("is_zero",s)){
	data_stream >> s;
	the_parameters.is_zero = string2svmfloat(s);
	if(the_parameters.is_zero <= 0){
	  throw read_exception("Invalid value for parameter 'is_zero'");
	};
      }
      else if(0 == strcmp("kernel_cache",s)){
	data_stream >> the_parameters.kernel_cache;
	if(the_parameters.kernel_cache < 0){
	  throw read_exception("Invalid value for parameter 'kernel_cache'");
	};
      }
      else if(0 == strcmp("convergence_epsilon",s)){
	data_stream >> s;
	the_parameters.convergence_epsilon = string2svmfloat(s);
	if(the_parameters.convergence_epsilon < 0){
	  throw read_exception("Invalid value for parameter 'convergence_epsilon'");
	};
      }
      else if(0 == strcmp("verbosity",s)){
	data_stream >> the_parameters.verbosity;
      }
      else if(0 == strcmp("min_prediction",s)){
	data_stream >> the_parameters.min_prediction;
	the_parameters.use_min_prediction = 1;
      }
      else if(0 == strcmp("print_w",s)){
        the_parameters.print_w = 1;
      }
      else if(0 == strcmp("loo_estim",s)){
        the_parameters.loo_estim = 1;
      }
      else if(0 == strcmp("no_loo_estim",s)){
        the_parameters.loo_estim = 0;
      }
      else if(0 == strcmp("format",s)){
	// default examples format
	data_stream >> s;
	if(0==strcmp("sparse",s)){
	  the_parameters.default_example_format.sparse = 1;
	}
	else{
	  the_parameters.default_example_format.sparse = 0;
	  the_parameters.default_example_format.where_x = 0;
	  the_parameters.default_example_format.where_y = 0;
	  the_parameters.default_example_format.where_alpha = 0;
	  for(int i=0;s[i] != '\0';i++){
	    if('x' == s[i]){
	      the_parameters.default_example_format.where_x = i+1;
	    }
	    else if('y' == s[i]){
	      the_parameters.default_example_format.where_y = i+1;
	    }
	    else if('a' == s[i]){
	      the_parameters.default_example_format.where_alpha = i+1;
	    }
	    else{
	      throw read_exception("Invalid default format for examples");
	    };
	  };
	  if(0 == the_parameters.default_example_format.where_x){
	    throw read_exception("Invalid default format for examples: x must be given");
	  };
	};
      }
      else if(0 == strcmp("delimiter",s)){
	// default examples delimiter
	data_stream >> s;
	if((s[0] != '\0') && (s[1] != '\0')){
	  the_parameters.default_example_format.delimiter = s[1];
	}
	else if ((s[1] == '\0') && (s[0] != '\0')){
	  the_parameters.default_example_format.delimiter = s[0];
	  if(' ' == data_stream.peek()){
	    // if delimiter = ' ' we have only read one '
	    data_stream.get();
	    if(the_parameters.default_example_format.delimiter == data_stream.peek()){
	      data_stream.get();
	      the_parameters.default_example_format.delimiter = ' ';
	    };
	  };
	}
	else{
	  the_parameters.default_example_format.delimiter = ' ';
	};
      }
      else{
	// unknown parameter
	char* t = new char[MAXCHAR];
	strcpy(t,"Unknown parameter: ");
	t = strcat(t,s);
	throw read_exception(t);
      };
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };

  if(the_parameters.is_pattern){
    the_parameters.do_scale_y = 0;
  };

  delete []s;
  return data_stream;
};


std::ostream& operator << (std::ostream& data_stream, parameters_c& the_parameters){
  data_stream<<"C "<<the_parameters.realC<<std::endl;
  if(1 == the_parameters.balance_cost){
    data_stream<<"balance_cost"<<std::endl;
  }
  else{
    if((1 != the_parameters.Lpos) || (1 != the_parameters.Lneg)){
      data_stream<<"L+ "<<the_parameters.Lpos<<std::endl;
      data_stream<<"L- "<<the_parameters.Lneg<<std::endl;
    };
  };
  if(the_parameters.epsilon_pos == the_parameters.epsilon_neg){
    data_stream<<"epsilon "<<the_parameters.epsilon_pos<<std::endl;
  }
  else{
    data_stream<<"epsilon+ "<<the_parameters.epsilon_pos<<std::endl;
    data_stream<<"epsilon- "<<the_parameters.epsilon_neg<<std::endl;
  };
  if(1 == the_parameters.is_distribution){
    data_stream<<"distribution"<<std::endl;
  };
  if(1 == the_parameters.is_nu){
    data_stream<<"nu "<<the_parameters.nu<<std::endl;
  };
  if(the_parameters.quadraticLossPos)
    data_stream<<"quadraticLoss+"<<std::endl;
  if(the_parameters.quadraticLossNeg)
    data_stream<<"quadraticLoss-"<<std::endl;
  if(the_parameters.search_c != 'n'){
    data_stream<<"search_C "<<the_parameters.search_c<<std::endl
	       <<"search_stop "<<the_parameters.search_stop<<std::endl
	       <<"Cmin "<<the_parameters.c_min<<std::endl
	       <<"Cmax "<<the_parameters.c_max<<std::endl
	       <<"Cdelta "<<the_parameters.c_delta<<std::endl;
  };
  if(the_parameters.is_pattern)
    data_stream<<"pattern"<<std::endl;
  if(the_parameters.do_scale)
    data_stream<<"scale"<<std::endl;
  if(the_parameters.cross_validation > 0)
    data_stream<<"cross_validation "<<the_parameters.cross_validation<<std::endl;
  if(the_parameters.cv_window>0)
    data_stream<<"cv_window"<<the_parameters.cv_window<<std::endl;
  if(the_parameters.cv_inorder)
    data_stream<<"cv_inorder"<<std::endl;
  data_stream<<"# optimization parameters"<<std::endl;
  data_stream<<"working_set_size "<<the_parameters.working_set_size<<std::endl;
  data_stream<<"max_iterations "<<the_parameters.max_iterations<<std::endl;
  data_stream<<"shrink_const "<<the_parameters.shrink_const<<std::endl;
  data_stream<<"descend "<<the_parameters.descend<<std::endl;
  data_stream<<"convergence_epsilon "<<the_parameters.convergence_epsilon<<std::endl;
  data_stream<<"kernel_cache "<<the_parameters.kernel_cache<<std::endl;
  data_stream<<"verbosity "<<the_parameters.verbosity<<std::endl;

  return data_stream;
};
