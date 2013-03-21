#include "kernel.h"


/**
 *
 * kernel_container_c
 *
 **/

kernel_container_c::~kernel_container_c(){
  kernel = 0;
};


kernel_c* kernel_container_c::get_kernel(){
  if(kernel == 0){
    kernel = new kernel_dot_c();
    is_linear=1;    
  };
  return kernel;
};


void kernel_container_c::clear(){
  // do not delete kernel, for reading of aggregation kernels
  kernel = 0;
};


std::istream& operator >> (std::istream& data_stream, kernel_container_c& the_kernel){
  char* s = new char[MAXCHAR];

  if(data_stream.eof() || ('@' == data_stream.peek())){
    // no kernel definition, take dot as default
    if(0 != the_kernel.kernel){
      delete the_kernel.kernel;
    };
    the_kernel.kernel = new kernel_dot_c();
    //    throw read_exception("No kernel definition found");
  }
  else{
    while((! data_stream.eof()) &&
	  (('#' == data_stream.peek()) ||
	   ('\n' == data_stream.peek()))){
	// ignore comment & newline
	data_stream.getline(s,MAXCHAR);
    };
    data_stream >> s;
    if(0 == strcmp("type",s)) {
      the_kernel.is_linear=0;
      data_stream >> s;
      if(0==strcmp("dot",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_dot_c();
	the_kernel.is_linear=1;
	data_stream >> *(the_kernel.kernel);
      }
      else if(0==strcmp("lin_dot",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_lin_dot_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("polynomial",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_polynomial_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("radial",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_radial_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("neural",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_neural_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("anova",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_anova_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("fourier",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_fourier_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("reg_fourier",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_reg_fourier_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("exponential",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_exponential_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if((0==strcmp("complete_matrix",s))||(0==strcmp("comp",s))){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_regularized_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if((0==strcmp("regularized",s))||(0==strcmp("reg",s))){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_regularized_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if((0==strcmp("aggregation",s))||(0==strcmp("sum_aggregation",s))){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_aggregation_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("prod_aggregation",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_prod_aggregation_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("zero",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_zero_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("lintransform",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_lintransform_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("user",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_user_c();
	data_stream >> *(the_kernel.kernel);
      }
      else if(0 == strcmp("user2",s)){
	if(0 != the_kernel.kernel){
	  delete the_kernel.kernel;
	};
	the_kernel.kernel = new kernel_user2_c();
	data_stream >> *(the_kernel.kernel);

      }
      // insert code for other kernels here
      else{
	char* t = new char[MAXCHAR];
	strcpy(t,"Unknown kernel type: ");
	strcat(t,s);
	throw read_exception(t);
      };
    }
    else{
      std::cout<<"read: "<<s<<std::endl;
      throw read_exception("kernel type has to be defined first");
    };
  };
  delete []s;
  return data_stream;
};


std::ostream& operator << (std::ostream& data_stream, kernel_container_c& the_kernel){
  if(0 != the_kernel.kernel){
    data_stream << *(the_kernel.kernel);
  }
  else{
    data_stream << "Empty kernel"<<std::endl;
  };
  return data_stream;
};


/**
 *
 * kernel_c
 *
 **/
kernel_c::kernel_c(){
  dim =0;
  cache_size=0;
  examples_size = 0;
  rows = 0;
  last_used = 0;
  index = 0;
  counter=0;
  // cache profiling:
  //  cache_misses = 0;
  //  cache_access = 0;
};

kernel_c::~kernel_c(){
  //  cout<<"destructor"<<endl;
  // cache profiling:
  //  cout<<cache_access<<" access to the cache"<<endl;
  //  cout<<cache_misses<<" cache misses ("<<100.0*(SVMFLOAT)cache_misses/((SVMFLOAT)cache_access)<<"%)"<<endl;
  clean_cache();
};

SVMFLOAT kernel_c::calculate_K(const SVMINT i, const SVMINT j){
  //  cout<<"K("<<i<<","<<j<<")"<<endl;
  //    if(cached(i) && cached(j)){
  //      // both are cached -> not shrinked
  //      return(rows[lookup(i)][lookup(j)]);
  //    };
//   SVMINT pos_x = lookup(i);
//   SVMINT pos_y = lookup(j);
//   if((index[pos_x] == i) && (index[pos_y] == j)
//      && (last_used[pos_x] != 0) && (last_used[pos_y] != 0)){
//     return rows[pos_x][j];
//   };
  svm_example x = the_examples->get_example(i);
  svm_example y = the_examples->get_example(j);
  return(calculate_K(x,y));
};

inline
SVMFLOAT kernel_c::calculate_K(const svm_example x, const svm_example y){
  // default is inner product
  return innerproduct(x,y); 
};

inline
SVMFLOAT kernel_c::innerproduct(const svm_example x, const svm_example y){
  // returns x*y
  SVMFLOAT result=0;

  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  svm_attrib* length_x = &(att_x[x.length]);
  svm_attrib* length_y = &(att_y[y.length]);

  while((att_x < length_x) && (att_y < length_y)){
    if(att_x->index == att_y->index){
      result += (att_x->att)*(att_y->att);
      att_x++;
      att_y++;
    }
    else if(att_x->index < att_y->index){
      att_x++;
    }
    else{
      att_y++;
    };
  };

  return result;
};


int kernel_c::cached(const SVMINT i){
  int ok;
  SVMINT pos = lookup(i);
  if(index[pos] == i){
    if(last_used[pos] > 0){
      ok = 1;
    }
    else{
      ok = 0;
    };
  }
  else{
    ok = 0;
  };
  return(ok);
};


SVMFLOAT kernel_c::norm2(const svm_example x, const svm_example y){
  // returns ||x-y||^2
  SVMFLOAT result=0;
  SVMINT length_x = x.length;
  SVMINT length_y = y.length;
  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  SVMINT pos_x=0;
  SVMINT pos_y=0;
  SVMFLOAT dummy;
  while((pos_x < length_x) && (pos_y < length_y)){
    if(att_x[pos_x].index == att_y[pos_y].index){
      dummy = att_x[pos_x++].att-att_y[pos_y++].att;
      result += dummy*dummy;
    }
    else if(att_x[pos_x].index < att_y[pos_y].index){
      dummy = att_x[pos_x++].att;
      result += dummy*dummy;
    }
    else{
      dummy = att_y[pos_y++].att;
      result += dummy*dummy;
    };
  };
  while(pos_x < length_x){
    dummy = att_x[pos_x++].att;
    result += dummy*dummy;
  };
  while(pos_y < length_y){
    dummy = att_y[pos_y++].att;
    result += dummy*dummy;
  };
  return result;
};


int kernel_c::check(){
  // check cache integrity, for debugging
  int result = 1;

  std::cout<<"Checking cache"<<std::endl;
  SVMINT i;
  // rows != 0
  for(i=0;i<cache_size;i++){
    if(rows[i] == 0){
      std::cout<<"ERROR: row["<<i<<"] = 0"<<std::endl;
      result = 0;
    };
  };
  std::cout<<"rows[i] checked"<<std::endl;

  // 0 <= index <= examples_size
  if(index != 0){
    SVMINT last_i=index[0];
    for(i=0;i<=cache_size;i++){
      if(index[i]<0){
	std::cout<<"ERROR: index["<<i<<"] = "<<index[i]<<std::endl;
	result = 0;
      };
      if(index[i]>examples_size){
	std::cout<<"ERROR: index["<<i<<"] = "<<index[i]<<std::endl;
	result = 0;
      };
      if(index[i]<last_i){
	std::cout<<"ERROR: index["<<i<<"] descending"<<std::endl;
	result = 0;
      };
      last_i = index[i];
    };
  };
  std::cout<<"index[i] checked"<<std::endl;

  // 0 <= last_used <= counter
  for(i=0;i<cache_size;i++){
    if(last_used[i]<0){
      std::cout<<"ERROR: last_used["<<i<<"] = "<<last_used[i]<<std::endl;
      result = 0;
    };
    if(last_used[i]>counter){
      std::cout<<"ERROR: last_used["<<i<<"] = "<<last_used[i]<<std::endl;
      result = 0;
    };
  };
  std::cout<<"last_used[i] checked"<<std::endl;

  std::cout<<"complete cache test"<<std::endl;
  SVMFLOAT* adummy;
  SVMINT i2;
  for(i2=0;i2<cache_size;i2++){
    std::cout<<i2<<" "; std::cout.flush();
    adummy = new SVMFLOAT[examples_size];
    for(SVMINT ai=0;ai<examples_size;ai++) adummy[ai] = (rows[i2])[ai];
    delete [](rows[i2]);
    rows[i] = adummy;
  }
  std::cout<<"cache test succeeded"<<std::endl;



  return result;
};


void kernel_c::init(SVMINT cache_MB, example_set_c* new_examples){
//   cout<<"init"<<endl;
//   cout<<"cache_size = "<<cache_size<<endl;
//   cout<<"examples_size = "<<examples_size<<endl;
//   cout<<"rows = "<<rows<<endl;
//   if(rows != 0)cout<<"rows[0] = "<<rows[0]<<endl;
  clean_cache();
  the_examples = new_examples; 
  dim = the_examples->get_dim();
  cache_mem = cache_MB*1048576;
  // check if reserved memory big enough
  if(cache_mem<(SVMINT)(sizeof(SVMFLOAT)*the_examples->size()+sizeof(SVMFLOAT*)+2*sizeof(SVMINT))){
    // not enough space for one example, increaee
    cache_mem = sizeof(SVMFLOAT)*the_examples->size()+sizeof(SVMFLOAT*)+2*sizeof(SVMINT);
  };
  set_examples_size(the_examples->size());
};


void kernel_c::clean_cache(){
  counter=0;
  if(rows != 0){
    SVMINT i;
    for(i=0;i<cache_size;i++){
      if(0 != rows[i]){
	delete [](rows[i]);
	rows[i]=0;
      };
    };
    delete []rows;
  };
  if(last_used != 0) delete []last_used;
  if(index != 0) delete []index;
  rows=0;
  last_used=0;
  index=0;
  cache_size=0;
  examples_size=0;
};


inline
SVMINT kernel_c::lookup(const SVMINT i){
  // find row i in cache
  // returns pos of element i if i in cache,
  // returns pos of smallest element larger than i otherwise
  SVMINT low;
  SVMINT high;
  SVMINT med;

  low=0;
  high=cache_size;
  // binary search
  while(low<high){
    med = (low+high)/2;
    if(index[med]>=i){
      high=med;
    }
    else{
      low=med+1;
    };
  };
  return high;
};


void kernel_c::overwrite(const SVMINT i, const SVMINT j){
  // overwrite entry i with entry j
  // WARNING: only to be used for shrinking!

  // i in cache?
  SVMINT pos_i=lookup(i);
  SVMINT pos_j=lookup(j);

  if((index[pos_i] == i) && (index[pos_j] == j)){
    // swap pos_i and pos_j
    SVMFLOAT* dummy = rows[pos_i];
    rows[pos_i] = rows[pos_j];
    rows[pos_j] = dummy;
    last_used[pos_i] = last_used[pos_j];
  }
  else{
    // mark rows as invalid
    if(index[pos_i] == i){
      last_used[pos_i] = 0;
    }
    else if(index[pos_j] == j){
      last_used[pos_j] = 0;
    };
  };

  // swap i and j in all rows
  SVMFLOAT* my_row;
  for(pos_i=0;pos_i<cache_size;pos_i++){
    my_row = rows[pos_i];
    if(my_row != 0){
      my_row[i] = my_row[j];
    };
  };
};


void kernel_c::set_examples_size(const SVMINT new_examples_size){
  // cache row with new_examples_size entries only
  
  // cout<<"shrinking from "<<examples_size<<" to "<<new_examples_size<<endl;
  if(new_examples_size>examples_size){
    clean_cache();
    examples_size = new_examples_size;
    cache_size = cache_mem/(sizeof(SVMFLOAT)*examples_size+sizeof(SVMFLOAT*)+2*sizeof(SVMINT));
    if(cache_size>examples_size){
      cache_size = examples_size;
    };
    // init 
    rows = new SVMFLOAT*[cache_size];
    last_used = new SVMINT[cache_size];
    index = new SVMINT[cache_size+1];
    SVMINT i;
    for(i=0;i<cache_size;i++){
      rows[i] = 0; // new SVMFLOAT[new_examples_size];
      last_used[i] = 0;
      index[i] = new_examples_size;
    };
    index[cache_size] = new_examples_size;
  }
  else if(new_examples_size<examples_size){
    // copy as much rows into new cache as possible
    SVMINT old_cache_size=cache_size;
    cache_size = cache_mem/(sizeof(SVMFLOAT)*new_examples_size+sizeof(SVMFLOAT*)+2*sizeof(SVMINT));
    if(cache_size > new_examples_size){
      cache_size = new_examples_size;
    };
    if(cache_size>=old_cache_size){
      // skip it, enough space available
      cache_size=old_cache_size;
      return;
    };

    SVMFLOAT** new_rows = new SVMFLOAT*[cache_size];
    SVMINT* new_last_used = new SVMINT[cache_size];
    SVMINT* new_index = new SVMINT[cache_size+1];
    SVMINT old_pos=0;
    SVMINT new_pos=0;
    new_index[cache_size] = new_examples_size;
    while((old_pos<old_cache_size) && (new_pos < cache_size)){
      if(last_used[old_pos] > 0){
	// copy example into new cache at new_pos
	new_rows[new_pos] = new SVMFLOAT[new_examples_size];
	SVMINT j;
	for(j=0;j<new_examples_size;j++){
	  (new_rows[new_pos])[j] = (rows[old_pos])[j];
	};
	delete [](rows[old_pos]);
	new_last_used[new_pos] = last_used[old_pos];
	new_index[new_pos] = index[old_pos];
	new_pos++;
      }
      else{
	if(rows[old_pos] != 0){
	  delete [](rows[old_pos]);
	};
      };
      old_pos++;
    };
    while(new_pos < cache_size){
      new_rows[new_pos] = 0; //new SVMFLOAT[new_examples_size];
      new_last_used[new_pos] = 0;
      new_index[new_pos] = new_examples_size;
      new_pos++;
    };
    while(old_pos < old_cache_size){
      if(rows[old_pos] != 0){
	delete [](rows[old_pos]);
      };
      old_pos++;
    };
    delete []rows;
    rows = new_rows;
    delete []last_used;
    last_used = new_last_used;
    delete []index;
    index = new_index;
    examples_size = new_examples_size;
  };

};


void kernel_c::compute_row(const SVMINT i, SVMFLOAT* myrow){
  // place row i in row
  svm_example x = the_examples->get_example(i);
  svm_example y;
  SVMINT k;
  for(k=0;k<examples_size;k++){
    y = the_examples->get_example(k);
    myrow[k] = calculate_K(x,y);
  };
};


SVMFLOAT* kernel_c::get_row(const SVMINT i){
  // lookup row in cache or compute
  SVMINT low=0;
  SVMINT high=cache_size;
  SVMINT pos=0;
  SVMINT j;
  // binary search for i in [low,high]
  high = lookup(i);
  if(high==cache_size){
    pos = high-1;
  }
  else{
    pos=high;
  };
  if((index[pos] != i) || (last_used[pos] == 0)){
    // cache miss
    SVMINT k;
    if(index[pos] == i){
      low = pos;
    }
    else{
      SVMINT min_time = last_used[cache_size-1];  // empty entries are at the end
      low=cache_size-1;
      for(k=0;k<cache_size;k++){
	// search for last recently used element
	if(last_used[k] < min_time){
	  min_time = last_used[k];
	  low = k;
	};
      };
    };
    
    // delete low, calculate row i, place in high
    SVMFLOAT* a_row = rows[low];
    if(high<=low){
      for(j=low;j>high;j--){
	rows[j] = rows[j-1];
	index[j] = index[j-1];
	last_used[j] = last_used[j-1];
      };
    }
    else{
      for(j=low;j<high-1;j++){
	rows[j] = rows[j+1];
	index[j] = index[j+1];
	last_used[j] = last_used[j+1];
      };
      high--;
    };
    pos=high;
    if(0 == a_row){
      a_row = new SVMFLOAT[examples_size];
    };
    rows[high] = a_row;
    compute_row(i,a_row);
    index[high]=i;
  };

  counter++;
  last_used[pos] = counter;

  return(rows[pos]);
};


void kernel_c::input(std::istream& data_stream){
  throw read_exception("ERROR: Attempt to read in abstract kernel.");
};

void kernel_c::output(std::ostream& data_stream) const{
  data_stream<<"Abstract kernel"<<std::endl;
};

std::istream& operator >> (std::istream& data_stream, kernel_c& the_kernel){
  the_kernel.input(data_stream);
  //  throw read_exception("ERROR: Attempt to read in abstract kernel.");
  return data_stream;
};

std::ostream& operator << (std::ostream& data_stream, kernel_c& the_kernel){
  the_kernel.output(data_stream);
  //  data_stream<<"Abstract kernel"<<endl;
  return data_stream;
};


/*
 *
 * The following kernels are defined
 * - kernel_dot_c: inner product
 * - kernel_pol_c: polynomial
 * - kernel_radial_c: radial basis function
 * plus:
 * - kernel_user_c: user defined kernel 1
 * - kernel_user2_c: user defined kernel 2
 *
 */

/*
 *
 * kernel_dot_c
 *
 */
SVMFLOAT kernel_dot_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result = innerproduct(x,y);
  return(result);
};

void kernel_dot_c::input(std::istream& data_stream){
  // read comments until next @, throw error at parameters
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };

  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if((next != '#') && (next != '\n')){
      // trying to read in parameter
      std::cout<<"WARNING: Parameters for dot kernel are ignored."<<std::endl;
    };
    data_stream.getline(s,MAXCHAR);
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};

void kernel_dot_c::output(std::ostream& data_stream) const{
  data_stream<<"type dot"<<std::endl;
};


/*
 *
 * kernel_lin_dot_c
 *
 */
SVMFLOAT kernel_lin_dot_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result = a*innerproduct(x,y)+b;
  return(result);
};


void kernel_lin_dot_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  a=1; 
  b=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("a",s)){
	data_stream >> a;
      }
      if(0 == strcmp("b",s)){
	data_stream >> b;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};


void kernel_lin_dot_c::output(std::ostream& data_stream) const{
  data_stream<<"type dot"<<std::endl;
  data_stream<<"a "<<a<<std::endl;
  data_stream<<"b "<<b<<std::endl;
};


/*
 *
 * kernel_polynomial_c
 *
 */
SVMFLOAT kernel_polynomial_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT prod=1+innerproduct(x,y);
  SVMFLOAT result=1;
  SVMINT i;
  for(i=0;i<degree;i++) result *= prod;
  return (result);
};

void kernel_polynomial_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("degree",s)){
	data_stream >> degree;
	ok = 1;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(! ok){
    throw read_exception("The parameters did not contain a valid description of a polynomial kernel.");
  };
  delete []s;
};

void kernel_polynomial_c::output(std::ostream& data_stream) const{
  data_stream<<"type polynomial"<<std::endl;
  data_stream<<"degree "<<degree<<std::endl;
};

/*
 *
 * kernel_radial_c
 *
 */
SVMFLOAT kernel_radial_c::calculate_K(const svm_example x, const svm_example y){
  return exp(-gamma*norm2(x,y));
};

void kernel_radial_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("gamma",s)){
	data_stream >> gamma;
	if(gamma <= 0){
	  throw read_exception("ERROR: Gamma must be > 0.");
	};
	ok = 1;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(! ok){
    throw read_exception("The parameters did not contain a valid description of a radial kernel.");
  };
  delete []s;
};

void kernel_radial_c::output(std::ostream& data_stream) const{
  data_stream<<"type radial"<<std::endl;
  data_stream<<"gamma "<<gamma<<std::endl;
};

/*
 *
 * kernel_neural_c
 *
 */
SVMFLOAT kernel_neural_c::calculate_K(const svm_example x, const svm_example y){
  return tanh(a*innerproduct(x,y)+b);
};

void kernel_neural_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  a=1; 
  b=1;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("a",s)){
	data_stream >> a;
      }
      else if(0 == strcmp("b",s)){
	data_stream >> b;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};

void kernel_neural_c::output(std::ostream& data_stream) const{
  data_stream<<"type neural"<<std::endl;
  data_stream<<"a "<<a<<std::endl;
  data_stream<<"b "<<b<<std::endl;
};

/*
 *
 * kernel_anova_c
 *
 */
SVMFLOAT kernel_anova_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=0;
  SVMINT length_x = x.length;
  SVMINT length_y = y.length;
  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  SVMINT pos_x=0;
  SVMINT pos_y=0;
  SVMINT zeros=dim;
  SVMFLOAT diff;
  while((pos_x < length_x) && (pos_y < length_y)){
    if(att_x[pos_x].index == att_y[pos_y].index){
      diff = att_x[pos_x++].att-att_y[pos_y++].att;
      result += exp(-gamma*(diff*diff));
    }
    else if(att_x[pos_x].index < att_y[pos_y].index){
      diff = att_x[pos_x++].att;
      result += exp(-gamma*(diff*diff));
    }
    else{
      diff = att_y[pos_y++].att;
      result += exp(-gamma*(diff*diff));
    };
    zeros--;
  };
  while(pos_x < length_x){
    diff = att_x[pos_x++].att;
    result += exp(-gamma*(diff*diff));
    zeros--;
  };
  while(pos_y < length_y){
    diff = att_y[pos_y++].att;
    result += exp(-gamma*(diff*diff));
    zeros--;
  };
  result += (SVMFLOAT)zeros;
  SVMFLOAT result2=1;
  SVMINT i;
  for(i=0;i<degree;i++){
    result2 *= result;
  };
  return result2;
};

void kernel_anova_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok_gamma=0;
  int ok_degree=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("gamma",s)){
	data_stream >> gamma;
	ok_gamma = 1;
      }
      else if(0 == strcmp("degree",s)){
	data_stream >> degree;
	ok_degree = 1;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if((!ok_gamma) || (!ok_degree)){
    throw read_exception("The parameters did not contain a valid description of an anova kernel.");
  };
  delete []s;
};

void kernel_anova_c::output(std::ostream& data_stream) const{
  data_stream<<"type anova"<<std::endl;
  data_stream<<"gamma "<<gamma<<std::endl;
  data_stream<<"degree "<<degree<<std::endl;
};


/*
 *
 * kernel_exponential_c
 *
 */ 
SVMFLOAT kernel_exponential_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=0;
  SVMINT length_x = x.length;
  SVMINT length_y = y.length;
  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  SVMINT pos_x=0;
  SVMINT pos_y=0;
  SVMINT i=0;
  SVMFLOAT mylambda=1;
  while((pos_x < length_x) && (pos_y < length_y)){
    if(att_x[pos_x].index == att_y[pos_y].index){
      for(;i<att_x[pos_x].index;i++) mylambda *= lambda;
      result += mylambda*att_x[pos_x++].att*att_y[pos_y++].att;
    }
    else if(att_x[pos_x].index < att_y[pos_y].index){
      pos_x++;
    }
    else{
      pos_y++;
    };
  };
  return result;
};

void kernel_exponential_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok_lambda=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("lambda",s)){
	data_stream >> lambda;
	ok_lambda = 1;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(!ok_lambda){
    throw read_exception("The parameters did not contain a valid description of an exponential kernel.");
  };
  delete []s;
};

void kernel_exponential_c::output(std::ostream& data_stream) const{
  data_stream<<"type exponential"<<std::endl;
  data_stream<<"lambda "<<lambda<<std::endl;
};


/*
 *
 *
 * kernel_aggregation_c : kernels, that consist of some other kernels
 *
 */

kernel_aggregation_c::kernel_aggregation_c(){
  number_elements = 0;
  elements = 0;
  from = 0;
  to = 0;
  new_x.example = 0;
  new_y.example = 0;
};


kernel_aggregation_c::~kernel_aggregation_c(){
  if(number_elements > 0){
    delete []elements;
    elements = 0;
    delete []from;
    from = 0;
    delete[] to;
    to = 0;
    number_elements = 0;
  };
  if(new_x.example){ delete []new_x.example; };
  if(new_y.example){ delete []new_y.example; };
};


void kernel_aggregation_c::init(SVMINT new_cache_MB,example_set_c* new_examples){
  kernel_c::init(new_cache_MB,new_examples);
  SVMINT i;
  for(i=0;i<number_elements;i++){
    (elements[i])->init(0,new_examples);
    (elements[i])->dim = to[i]-from[i];
  };
  new_x.example = new svm_attrib[dim];
  new_y.example = new svm_attrib[dim];
};


void kernel_aggregation_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  // WARNING: no checks of the input values are performed
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  char* s = new char[MAXCHAR];
  SVMINT parts_read=0;
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    (' ' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    ('\t' == next)){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("number_parts",s)){
	if(number_elements>0){
	  throw read_exception("Parameter 'number_parts' cannot be defined twice.");
	};
	data_stream >> number_elements;
	if(number_elements<=0){
	  throw read_exception("Invalid value for parameter 'number_parts'.");
	};
	elements = new kernel_c*[number_elements];
	from = new SVMINT[number_elements];
	to = new SVMINT[number_elements];
      }
      else if(0==strcmp("range",s)){
	if(parts_read<number_elements){
	  data_stream >> from[parts_read];
	  from[parts_read]--;
	  data_stream >> to[parts_read];
	  parts_read++;
	}
	else{
	  throw read_exception("too much ranges defined in aggregation kernel or 'number_parts' not given.");
	};
      }
      else{
	throw read_exception("Unknown parameter in aggregation kernel.");
      };
      data_stream.getline(s,MAXCHAR); // ignore rest of line
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(!data_stream.eof()){
    kernel_container_c cont;
    SVMINT i;
    for(i=0;i<number_elements;i++){
      // next line should be "@kernel"
      data_stream >> s;
      if(0==strcmp("@kernel",s)){
	data_stream.getline(s,MAXCHAR); // ignore rest of line
	data_stream >> cont;
	elements[i] = cont.get_kernel();
	cont.clear();
      }
      else{
	throw read_exception("Could not find enough kernel parts for aggregation kernel.");
      };
    };
  };
  delete []s;
};


void kernel_aggregation_c::output_aggregation(std::ostream& data_stream) const{
  data_stream<<"number_parts "<<number_elements<<std::endl;
  SVMINT i;
  for(i=0;i<number_elements;i++){
    data_stream<<"range "
	       <<(from[i]+1)  // inner format: [from,to[, starting at 0
                              // io format:    [from,to], starting at 1
	       <<" "<<to[i]<<std::endl;
  };
  for(i=0;i<number_elements;i++){
    data_stream<<"@kernel"<<std::endl
	       <<"# "<<(i+1)<<". part of aggregation kernel"<<std::endl;
    (elements[i])->output(data_stream);
  };
};


void kernel_aggregation_c::output(std::ostream& data_stream)const{
  data_stream<<"type aggregation"<<std::endl;
  output_aggregation(data_stream);
};


SVMFLOAT kernel_aggregation_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=0;
  //  svm_example  new_x = x;
  //  svm_example new_y = y;
  SVMINT start;
  SVMINT stop;
  SVMINT count;

  for(SVMINT i=0;i<number_elements;i++){
    // find matching part of x and y
    start=0;
    while((start<x.length) && (((x.example)[start]).index < from[i])){
      start++;
    };
    //    new_x.example = x.example + start;
    stop=start;
    count=0;
    while((stop<x.length) && (((x.example)[stop]).index < to[i])){
      (new_x.example)[count] = (x.example)[stop];
      ((new_x.example)[count]).index -= from[i]; 
      //      cout<<"x: "<<((new_x.example)[count]).index<<" --> "<<((new_x.example)[count]).att<<endl;
      count++;
      stop++;
    };
    new_x.length = stop-start;
    start=0;
    while((start<y.length) && (((y.example)[start]).index < from[i])){
      start++;
    };
    //    new_y.example = y.example + start;
    stop=start;
    count=0;
    while((stop<y.length) && (((y.example)[stop]).index < to[i])){
      (new_y.example)[count] = (y.example)[stop];
      ((new_y.example)[count]).index -= from[i]; 
      //      cout<<"y: "<<((new_y.example)[count]).index<<" --> "<<((new_y.example)[count]).att<<endl;
      count++;
      stop++;
    };
    new_y.length = stop-start;

    // default ist sum-kernel
    result += (elements[i])->calculate_K(new_x,new_y);
  };
  //  exit(1);
  return result;
};

/*
 *
 *
 * kernel_prod_aggregation_c : kernel, that consist of the 
 *                             prodcut of some other kernels
 *
 */

kernel_prod_aggregation_c::kernel_prod_aggregation_c(){
  number_elements = 0;
  elements = 0;
  from = 0;
  to = 0;
  new_x.example = 0;
  new_y.example = 0;
};


kernel_prod_aggregation_c::~kernel_prod_aggregation_c(){
  if(number_elements > 0){
    delete []elements;
    elements = 0;
    delete []from;
    from = 0;
    delete[] to;
    to = 0;
    number_elements = 0;
  };
  if(new_x.example){ delete []new_x.example; };
  if(new_y.example){ delete []new_y.example; };
};


void kernel_prod_aggregation_c::output(std::ostream& data_stream)const{
  data_stream<<"type prod_aggregation"<<std::endl;
  output_aggregation(data_stream);
};


SVMFLOAT kernel_prod_aggregation_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=1;
  //  svm_example new_x = x;
  //  svm_example new_y = y;
  SVMINT start;
  SVMINT stop;
  SVMINT count;
  SVMINT i;
  for(i=0;i<number_elements;i++){
    // find matching part of x and y
    start=0;
    while((start<x.length) && (((x.example)[start]).index < from[i])){
      start++;
    };
    //    new_x.example = x.example + start;
    stop=start;
    count=0;
    while((stop<x.length) && (((x.example)[stop]).index < to[i])){
      (new_x.example)[count] = (x.example)[stop];
      ((new_x.example)[count]).index -= from[i]; 
      count++;
      stop++;
    };
    new_x.length = stop-start;
    start=0;
    while((start<y.length) && (((y.example)[start]).index < from[i])){
      start++;
    };
    //    new_y.example = y.example + start;
    stop=start;
    count = 0;
    while((stop<y.length) && (((y.example)[stop]).index < to[i])){
      (new_y.example)[count] = (y.example)[stop];
      ((new_y.example)[count]).index -= from[i]; 
      count++;
      stop++;
    };
    new_y.length = stop-start;

    // default ist sum-kernel
    result *= (elements[i])->calculate_K(new_x,new_y);

  };
  return result;
};


/*
 *
 * kernel_fourier_c: Kernel generating fourier expansions
 *
 */

kernel_fourier_c::kernel_fourier_c(){
  N = 1;
};


void kernel_fourier_c::input(std::istream& data_stream){
    // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if((0 == strcmp("n",s)) || 
	 (0 == strcmp("N",s))){
	data_stream >> N;
	ok = 1;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(! ok){
    throw read_exception("The parameters did not contain a valid description of a fourier kernel.");
  };
  delete []s;
};


void kernel_fourier_c::output(std::ostream& data_stream) const{
  data_stream<<"type fourier"<<std::endl;
  data_stream<<"N "<<N<<std::endl;
};
  

SVMFLOAT kernel_fourier_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=1;  
  SVMINT length_x = x.length;
  SVMINT length_y = y.length;
  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  SVMINT pos_x=0;
  SVMINT pos_y=0;
  SVMINT zeros=dim;
  SVMFLOAT diff;
  SVMFLOAT dummy;
  while((pos_x < length_x) && (pos_y < length_y)){
    if(att_x[pos_x].index == att_y[pos_y].index){
      diff = att_x[pos_x++].att-att_y[pos_y++].att;
    }
    else if(att_x[pos_x].index < att_y[pos_y].index){
      diff = att_x[pos_x++].att;
    }
    else{
      diff = -att_y[pos_y++].att;
    };
    dummy = sin(diff/2);
    if(0 == dummy){
      dummy = 1/2+(SVMFLOAT)N;
    }
    else{
      dummy = sin((2*(SVMFLOAT)N+1)/2*diff) / dummy;
    };
    result *= dummy;
    zeros--;
  };
  while(pos_x < length_x){
    diff = att_x[pos_x++].att;
    dummy = sin(diff/2);
    if(0 == dummy){
      dummy = 1/2+(SVMFLOAT)N;
    }
    else{
      dummy = sin((2*(SVMFLOAT)N+1)/2*diff) / dummy;
    };
    result *= dummy;
    zeros--;
  };
  while(pos_y < length_y){
    diff = att_y[pos_y++].att;
    dummy = sin(diff/2);
    if(0 == dummy){
      dummy = 1/2+(SVMFLOAT)N;
    }
    else{
      dummy = sin((2*(SVMFLOAT)N+1)/2*diff) / dummy;
    };
    result *= dummy;
    zeros--;
  };

  SVMINT i;
  for(i=0;i<zeros;i++){
    result *=  1/2+(SVMFLOAT)N;
  };
  return result;
};


/*
 *
 * kernel_reg_fourier_c: Kernel generating regularized fourier expansions
 *
 */

kernel_reg_fourier_c::kernel_reg_fourier_c(){
  q = 0.5;
};


void kernel_reg_fourier_c::input(std::istream& data_stream){
    // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  int ok=0;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if('\n' == next){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if((0 == strcmp("q",s)) || 
	 (0 == strcmp("Q",s))){
	data_stream >> q;
	if((q>0) && (q<1)){
	  ok = 1;
	};
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(! ok){
    throw read_exception("The parameters did not contain a valid description of a regularized fourier kernel.");
  };
  delete []s;
};


void kernel_reg_fourier_c::output(std::ostream& data_stream) const{
  data_stream<<"type reg_fourier"<<std::endl;
  data_stream<<"q "<<q<<std::endl;
};
  

SVMFLOAT kernel_reg_fourier_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=1;  
  SVMINT length_x = x.length;
  SVMINT length_y = y.length;
  svm_attrib* att_x = x.example;
  svm_attrib* att_y = y.example;
  SVMINT pos_x=0;
  SVMINT pos_y=0;
  SVMINT zeros=dim;
  SVMFLOAT diff;
  SVMFLOAT q2 = q*q;
  while((pos_x < length_x) && (pos_y < length_y)){
    if(att_x[pos_x].index == att_y[pos_y].index){
      diff = att_x[pos_x++].att-att_y[pos_y++].att;
    }
    else if(att_x[pos_x].index < att_y[pos_y].index){
      diff = att_x[pos_x++].att;
    }
    else{
      diff = -att_y[pos_y++].att;
    };
    diff *= PI;
    result *= (1-q2)/(2*(1-2*q*cos(diff)+q2));
    zeros--;
  };
  while(pos_x < length_x){
    diff = att_x[pos_x++].att;
    diff *= PI;
    result *= (1-q2)/(2*(1-2*q*cos(diff)+q2));
    zeros--;
  };
  while(pos_y < length_y){
    diff = att_y[pos_y++].att;
    diff *= PI;
    result *= (1-q2)/(2*(1-2*q*cos(diff)+q2));
    zeros--;
  };

  q2 = (1+q)/(2*(1-q));
  SVMINT i;
  for(i=0;i<zeros;i++){
    result *= q2;
  };
  return result;
};


/*
 *
 * kernel_zero_c: returns 0, dummy kernel
 *
 */
SVMFLOAT kernel_zero_c::calculate_K(const svm_example x, const svm_example y){
  return 0;
};


void kernel_zero_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    // ignore all lines
    data_stream.getline(s,MAXCHAR);
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};


void kernel_zero_c::output(std::ostream& data_stream) const{
  data_stream<<"type zero"<<std::endl;
};


/*
 *
 * kernel_lintransform_c: K = a*K'+b
 *
 */
kernel_lintransform_c::kernel_lintransform_c(){
  subkernel = 0;
  a=1;
  b=0;
};


kernel_lintransform_c::~kernel_lintransform_c(){
  if(0 != subkernel){
    delete subkernel;
  };
  subkernel = 0;
};


void kernel_lintransform_c::output(std::ostream& data_stream)const{
  data_stream<<"type lintransform"<<std::endl;
  data_stream<<"a "<<a<<std::endl;
  data_stream<<"b "<<b<<std::endl;
  data_stream<<"@kernel"<<std::endl;
  data_stream<<"# subkernel of lintransform kernel"<<std::endl;
  if(0 != subkernel){
    subkernel->output(data_stream);
  }
  else{
    data_stream<<"# not defined"<<std::endl;
    data_stream<<"type null"<<std::endl;
  };
};


void kernel_lintransform_c::input(std::istream& data_stream){
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    (' ' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    ('\t' == next)){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("a",s)){
	data_stream >> a;
	if(a<0){
	  throw read_exception("Invalid value for parameter 'a'.");
	};
      }
      else if(0 == strcmp("b",s)){
	data_stream >> b;
      }
      else{
	throw read_exception("Unknown parameter in lintransform kernel.");
      };
      data_stream.getline(s,MAXCHAR); // ignore rest of line
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(!data_stream.eof()){
    kernel_container_c cont;
    // next line should be "@kernel"
    data_stream >> s;
    if(0==strcmp("@kernel",s)){
      data_stream.getline(s,MAXCHAR); // ignore rest of line
      data_stream >> cont;
      subkernel = cont.get_kernel();
      cont.clear();
    }
    else{
      throw read_exception("Could not find subkernel for lintransform kernel.");
    };
  };
  delete []s;
};


SVMFLOAT kernel_lintransform_c::calculate_K(const svm_example x, const svm_example y){
  return (subkernel->calculate_K(x,y)*a+b);
};


/*
 *
 * kernel_regularized_c: regularize other kernel
 *
 */

kernel_regularized_c::kernel_regularized_c(){
  inner_kernel=0;
  cache=0;
};


kernel_regularized_c::~kernel_regularized_c(){
  if(inner_kernel){
    delete inner_kernel;
    inner_kernel=0;
  };
  if(cache){
    delete []cache;
    cache=0;
  };
};


void kernel_regularized_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  // WARNING: no checks of the input values are performed
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    (' ' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    ('\t' == next)){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      throw read_exception("Unknown parameter in regularized kernel.");
      data_stream.getline(s,MAXCHAR); // ignore rest of line
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  if(!data_stream.eof()){
    kernel_container_c cont;
    // next line should be "@kernel"
    data_stream >> s;
    if(0==strcmp("@kernel",s)){
      data_stream.getline(s,MAXCHAR); // ignore rest of line
      data_stream >> cont;
      inner_kernel = cont.get_kernel();
      cont.clear();
    }
    else{
      throw read_exception("Could not find inner kernel for regularized kernel.");
    };
  };
  delete []s;
};


void kernel_regularized_c::output(std::ostream& data_stream)const{
  data_stream<<"type regularized"<<std::endl;
  data_stream<<"@kernel"<<std::endl
	     <<"# inner kernel of regularized kernel"<<std::endl;
  inner_kernel->output(data_stream);
};


void kernel_regularized_c::init(SVMINT new_cache_MB,example_set_c* new_examples){
  kernel_c::init(new_cache_MB,new_examples);
  inner_kernel->init(0,new_examples);
  inner_kernel->dim = dim;
  if(cache){ delete []cache; };
  cache = new SVMFLOAT[examples_size];
  SVMINT i;
  svm_example x;
  for(i=0;i<examples_size;i++){
    x = the_examples->get_example(i);
    cache[i] = inner_kernel->calculate_K(x,x);
    if(cache[i] > 0){
      cache[i] = sqrt(cache[i]);
    }
    else{
      cache[i] = 0;
    };
  };
};


void kernel_regularized_c::compute_row(const SVMINT i, SVMFLOAT* myrow){
  // place row i in row
  svm_example x;
  svm_example y;
  SVMFLOAT res = cache[i];
  SVMFLOAT res2;
  SVMINT k;
  if(res <= 0){
    for(k=0;k<examples_size;k++){
      myrow[k] = 0;
    };
  }
  else{
    x = the_examples->get_example(i);
    for(k=0;k<examples_size;k++){
      if(k == i){
	myrow[k] = 1;
      }
      else{
	res2 = cache[k];
	if(res2 <= 0){
	  myrow[k] = 0;
	}
	else{
	  y = the_examples->get_example(k);
	  myrow[k] = inner_kernel->calculate_K(x, y)/(res*res2);
	};
      };
    };
  };
};


void kernel_regularized_c::overwrite(const SVMINT i, const SVMINT j){
  kernel_c::overwrite(i,j);
  SVMFLOAT tmp = cache[i];
  cache[i] = cache[j];
  cache[j] = tmp;
};


SVMFLOAT kernel_regularized_c::calculate_K(const svm_example x, const svm_example y){
  // use caching here!!!
  SVMFLOAT res = inner_kernel->calculate_K(x,x)*inner_kernel->calculate_K(y,y);
  if(res > 0){
    res = inner_kernel->calculate_K(x,y)/sqrt(res);
  }
  else{
    res=0;
  };
  return res;
};


/*
 *
 * kernel_user_c: Enter your own kernel code here
 *
 */ 
kernel_user_c::kernel_user_c(){
  param_i_1 = 0;
  param_i_2 = 0;
  param_i_3 = 0;
  param_i_4 = 0;
  param_i_5 = 0;
  param_f_1 = 0;
  param_f_2 = 0;
  param_f_3 = 0;
  param_f_4 = 0;
  param_f_5 = 0;
};


SVMFLOAT kernel_user_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=0;
  // this is where you can insert your own kernel calculation
  // you can use the SVMINT parameters param_i_1 ... param_i_5
  // and the SVMFLOAT parameters param_f_1 ... param_f_5

  // begin user defined kernel

  result = innerproduct(x,y);

  // end user defined kernel

  return result;
};

void kernel_user_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  // WARNING: no checks of the input values are performed
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    ('\t' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    (' ' == next)){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("param_i_1",s)){
	data_stream >> param_i_1;
      }
      else if(0 == strcmp("param_i_2",s)){
	data_stream >> param_i_2;
      }
      else if(0 == strcmp("param_i_3",s)){
	data_stream >> param_i_3;
      }
      else if(0 == strcmp("param_i_4",s)){
	data_stream >> param_i_4;
      }
      else if(0 == strcmp("param_i_5",s)){
	data_stream >> param_i_5;
      }
      else if(0 == strcmp("param_f_1",s)){
	data_stream >> param_f_1;
      }
      else if(0 == strcmp("param_f_2",s)){
	data_stream >> param_f_2;
      }
      else if(0 == strcmp("param_f_3",s)){
	data_stream >> param_f_3;
      }
      else if(0 == strcmp("param_f_4",s)){
	data_stream >> param_f_4;
      }
      else if(0 == strcmp("param_f_5",s)){
	data_stream >> param_f_5;
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};

void kernel_user_c::output(std::ostream& data_stream) const{
  data_stream<<"type user"<<std::endl;
  data_stream<<"param_i_1 "<<param_i_1<<std::endl;
  data_stream<<"param_i_2 "<<param_i_2<<std::endl;
  data_stream<<"param_i_3 "<<param_i_3<<std::endl;
  data_stream<<"param_i_4 "<<param_i_4<<std::endl;
  data_stream<<"param_i_5 "<<param_i_5<<std::endl;
  data_stream<<"param_f_1 "<<param_f_1<<std::endl;
  data_stream<<"param_f_2 "<<param_f_2<<std::endl;
  data_stream<<"param_f_3 "<<param_f_3<<std::endl;
  data_stream<<"param_f_4 "<<param_f_4<<std::endl;
  data_stream<<"param_f_5 "<<param_f_5<<std::endl;
};

/*
 *
 * kernel_user2_c: Enter your own kernel code here
 *
 */
kernel_user2_c::kernel_user2_c(){
  number_param = 100;
  param_i = new SVMINT[number_param];
  param_f = new SVMFLOAT[number_param];
}


kernel_user2_c::~kernel_user2_c(){
  if(param_i != 0) delete param_i;
  if(param_f != 0) delete param_f;
};


SVMFLOAT kernel_user2_c::calculate_K(const svm_example x, const svm_example y){
  SVMFLOAT result=0;
  // this is where you can insert your own kernel calculation
  // you can use the SVMINT parameters param_i[0] ... param_i[number_param-1]
  // and the SVMFLOAT parameters param_f[0] ... param_f[number_param-1]
  // number_param is defined in the class constructor

  // begin user defined kernel

  result = norm2(x,y);

  // end user defined kernel

  return result;
};

void kernel_user2_c::input(std::istream& data_stream){
  // read comments and parameters until next @
  // WARNING: no checks of the input values are performed
  char next = data_stream.peek();
  if(next == EOF){ 
    // set stream to eof
    next = data_stream.get(); 
  };
  SVMINT pos;
  char* s = new char[MAXCHAR];
  while((! data_stream.eof()) && (next != '@')){
    if('#' == next){
      // ignore comment
      data_stream.getline(s,MAXCHAR);
    }
    else if(('\n' == next) ||
	    ('\t' == next) ||
	    ('\r' == next) ||
	    ('\f' == next) ||
	    (' ' == next)){
      // ignore line-end
      next = data_stream.get();
    }
    else{
      // trying to read in parameter
      data_stream >> s;
      if(0 == strcmp("param_i",s)){
	data_stream >> pos;
	if((pos >= 0) && (pos<number_param)){
	  data_stream >> param_i[pos];
	}
	else{
	  throw read_exception("Illegal parameter index for param_i.");
	};
      }
      else if(0==strcmp("param_f",s)){
	data_stream >> pos;
	if((pos >= 0) && (pos<number_param)){
	  data_stream >> param_f[pos];
	}
	else{
	  throw read_exception("Illegal parameter index for param_f.");
	};
      }
      else{
	std::cout<<"Ignoring unknown parameter: "<<s<<std::endl;
      };
      data_stream.getline(s,MAXCHAR);
    };
    next = data_stream.peek();
    if(next == EOF){ 
      // set stream to eof
      next = data_stream.get(); 
    };
  };
  delete []s;
};

void kernel_user2_c::output(std::ostream& data_stream) const{
  data_stream<<"type user"<<std::endl;
  SVMINT i;
  for(i=0;i<number_param;i++){
    data_stream<<"param_i "<<i<<" "<<param_i[i]<<std::endl;
  };
  for(i=0;i<number_param;i++){
    data_stream<<"param_f "<<i<<" "<<param_f[i]<<std::endl;
  };
};
