#include "example_set.h"
#include <stdio.h>

example_set_c::example_set_c(){
  init(0,0);
};


example_set_c::example_set_c(SVMINT new_total, SVMINT new_dim){
  init(new_total,new_dim);
};



void example_set_c::init(SVMINT new_total, SVMINT new_dim){
  examples_total = 0;
  capacity=0;
  has_y = 0;
  has_alphas = 0;
  has_scale = 0;
  has_pattern_y = 1;
  b = 0;
  all_alphas = 0;
  all_ys = 0;
  // create dummy
  the_set = new svm_example[1];
  the_set[0].example = 0;
  dim=0;
  Exp = 0;
  Var = 0;
  filename = new char[MAXCHAR];
  filename[0]='\0';
  set_dim(new_dim),
  resize(capacity);
  // set default file format
  my_format.sparse = 0;
  my_format.where_x = 2;
  my_format.where_y = 1;
  my_format.where_alpha = 0;
  my_format.delimiter = ' ';
};


example_set_c::~example_set_c(){
  delete []filename;
  clear();
  if(the_set) delete []the_set;
};


void example_set_c::set_format(example_format new_format){
  my_format = new_format;
};


void example_set_c::set_filename(char* new_filename){
  strcpy(filename,new_filename);
};


void example_set_c::clear(){
  if(all_alphas){
    delete []all_alphas;
    all_alphas=0;
  };
  if(all_ys){
    delete []all_ys;
    all_ys=0;
  };
  if(the_set){
    SVMINT i;
    SVMINT c;
    for(i=0;i<capacity;i++){
      if(the_set[i].example != 0){

	for(c=0;c<the_set[i].length;c++){
	  (((the_set[i]).example)[c]).index=28;
	  (((the_set[i]).example)[c]).att = 4;
	};

	delete [](the_set[i].example);
	the_set[i].example=0;
      };
    };
    delete []the_set;
  };
  if(Exp) delete []Exp;
  if(Var) delete []Var;
  Exp = 0;
  Var = 0;
  the_set = new svm_example[1];
  the_set[0].example = 0;
  dim = 0;
  b = 0;
  examples_total = 0;
  capacity = 0;
  has_y = 0;
  has_alphas = 0;
  has_scale = 0;
};


SVMINT example_set_c::size(){
  return examples_total;
};


SVMINT example_set_c::size_pos(){
  SVMINT i;
  SVMINT count=0;
  for(i=0;i<capacity;i++){
    if(the_set[i].y > 0){
      count++;
    };
  };
  return count;
};


SVMINT example_set_c::size_neg(){
  SVMINT i;
  SVMINT count=0;
  for(i=0;i<capacity;i++){
    if(the_set[i].y < 0){
      count++;
    };
  };
  return count;
};


void example_set_c::set_dim(SVMINT new_dim){
  if(new_dim<dim){
    throw general_exception("ERROR: Trying to decrease dimension of examples");
  };
  dim = new_dim;
  if(Exp) delete []Exp;
  if(Var) delete []Var;
  Exp = 0;
  Var = 0;
};


SVMINT example_set_c::get_dim(){
  return dim;
};


void example_set_c::resize(SVMINT new_total){
  svm_example* new_set = 0;
  SVMINT i;
  if(new_total > capacity){
    // add new space to set
    new_set = new svm_example[new_total];
    // copy old values
    for(i=0;i<capacity;i++){
      new_set[i] = the_set[i];
    };
    for(i=capacity;i<new_total;i++){
      new_set[i].example = 0;
      new_set[i].length = 0;
    };
    delete []the_set;
    the_set=new_set;
    capacity = new_total;
    if(all_alphas != 0){
      delete []all_alphas;
      all_alphas = 0;
    };
    if(all_ys != 0){
      delete []all_ys;
      all_ys = 0;
    };
  }
  else if(new_total < capacity){
    new_set = new svm_example[new_total];
    // copy remaining values
    for(i=0;i<new_total;i++){
      new_set[i] = the_set[i];
    };
    // delete obsolete values
    for(i=new_total;i<capacity;i++){
      if(the_set[i].example != 0){
	delete [](the_set[i].example);
	the_set[i].example = 0;
	examples_total--;
      };
    };
    delete []the_set;
    the_set=new_set;
    capacity = new_total;

    if(all_alphas != 0){
      delete []all_alphas;
      all_alphas = 0;
    };
    if(all_ys != 0){
      delete []all_ys;
      all_ys = 0;
    };
  };
};


void example_set_c::swap(SVMINT i, SVMINT j){
  svm_example ex_dummy;
  SVMFLOAT dummy;
  ex_dummy=the_set[i];
  the_set[i] = the_set[j];
  the_set[j] = ex_dummy;
  if(all_alphas != 0){
    dummy=all_alphas[i];
    all_alphas[i] = all_alphas[j];
    all_alphas[j] = dummy;
  };
  if(all_ys != 0){
    dummy = all_ys[i];
    all_ys[i] = all_ys[j];
    all_ys[j] = dummy;
  };
};


void example_set_c::put_example(const SVMINT pos, const SVMFLOAT* example){
  // examples is SVMFLOAT-array 1..dim

  SVMINT non_zero=0;
  svm_attrib* new_att;
  SVMINT i;
  for(i=0;i<dim;i++){
    if(0 != example[i]){
      non_zero++;
    };
  };
  if(pos>=capacity){
    // make set bigger
    resize(2*capacity+1);
  };
  if(0 == the_set[pos].example){
    // add new example, reserve space for y and alpha
    examples_total++;
  }
  else{
    delete [](the_set[pos].example);
    the_set[pos].example = 0;
  };
  new_att = new svm_attrib[non_zero];
  the_set[pos].example = new_att;
  // add attributes
  SVMINT j=0;
  for(i=0;i<non_zero;i++){
    while(0 == example[j]) j++;
    new_att[i].att = example[j];
    new_att[i].index=j;
    j++;
  };
  the_set[pos].y = example[dim];
  the_set[pos].alpha = example[dim+1];
  the_set[pos].length = non_zero;
  if(all_alphas != 0){
    all_alphas[pos] = the_set[pos].alpha;
  };
  if(all_ys != 0){
    all_ys[pos]=the_set[pos].y;
  };
  if((the_set[pos].y != 1) && (the_set[pos].y != -1)){
    has_pattern_y = 0;
  };
};


void example_set_c::put_example(const SVMFLOAT* example){
  put_example(examples_total,example);
};


void example_set_c::put_example(const SVMINT pos, const svm_example example){
  if(pos>=capacity){
    // make set bigger
    resize(2*capacity+1);
  };
  if(the_set[pos].example != 0){
    // overwrite old
    delete [](the_set[pos].example);
    the_set[pos].example = 0;
  }
  else{
    examples_total++;
  };
  the_set[pos].length = example.length;
  svm_attrib* new_att = new svm_attrib[example.length];
  SVMINT i;
  for(i=0;i<example.length;i++){
    new_att[i] = example.example[i];
  };
  the_set[pos].example = new_att;
  the_set[pos].y = example.y;
  the_set[pos].alpha = example.alpha;
  if(all_alphas != 0){
    all_alphas[pos] = the_set[pos].alpha;
  };
  if(all_ys != 0){
    all_ys[pos] = the_set[pos].y;
  };
  if((the_set[pos].y != 1) && (the_set[pos].y != -1)){
    has_pattern_y = 0;
  };
};


void example_set_c::put_example(const svm_example example){
  put_example(examples_total,example);
};


svm_example example_set_c::get_example(const SVMINT pos){
  return the_set[pos];
};


void example_set_c::put_y(const SVMINT pos, const SVMFLOAT y){
  the_set[pos].y = y;
  if(all_ys != 0){
    all_ys[pos] = y;
  };
  if((y != 1) && (y != -1)){
    has_pattern_y = 0;
  };
};


SVMFLOAT example_set_c::get_y(const SVMINT pos){
  return (the_set[pos].y);
};


void example_set_c::put_alpha(const SVMINT pos, const SVMFLOAT alpha){
  the_set[pos].alpha = alpha;
  if(all_alphas != 0){
    all_alphas[pos] = alpha;
  };
};


SVMFLOAT example_set_c::get_alpha(const SVMINT pos){
  return (the_set[pos].alpha);
};


void example_set_c::put_b(const SVMFLOAT new_b){
  b=new_b;
};


SVMFLOAT example_set_c::get_b(){
  return b;
};


SVMFLOAT* example_set_c::get_alphas(){
  if(0 == all_alphas){
    SVMINT the_size = size();
    all_alphas = new SVMFLOAT[the_size];
    SVMINT i;
    for(i=0; i<the_size; i++){
      all_alphas[i] = get_alpha(i);
    };
  };
  return all_alphas;
};


SVMFLOAT* example_set_c::get_ys(){
  if(0 == all_ys){
    SVMINT the_size = size();
    all_ys = new SVMFLOAT[the_size];
    SVMINT i;
    for(i=0; i<the_size; i++){
      all_ys[i] = get_y(i);
    };
  };
  return all_ys;
};


void example_set_c::compress(){
  // remove zeros out of examples
  SVMINT next=0;
  SVMINT i=0;
  for(i=0;i<capacity;i++){
    if(the_set[i].example != 0){
      the_set[next] = the_set[i];
      if(next != i){
	the_set[i].example = 0;
	the_set[i].length = 0;
      };
      next++;
    };
  };
  resize(next);
};


void example_set_c::scale_alphas(const SVMFLOAT factor){
  // set alpha -> factor*alpha
  SVMINT i;
  for(i=0;i<capacity;i++){
    put_alpha(i,factor*get_alpha(i));
  };
  if(all_alphas){
    for(i=0;i<capacity;i++){
      all_alphas[i] = get_alpha(i);
    };
  };
};


SVMFLOAT example_set_c::get_y_var(){
  if(0 != Var ){
    if(0 != Var[dim]){
      return  Var[dim];
    };
  };
  return 1;
};

void example_set_c::scale(){
  scale(1);
};


void example_set_c::scale(int scale_y){
  if(examples_total == 0) return;

  if(Exp == 0) Exp = new SVMFLOAT[dim+1];
  if(Var == 0) Var = new SVMFLOAT[dim+1];

  SVMINT i;

  // calculate Exp and Var
  for(i=0;i<=dim;i++){
    Exp[i] = 0;
    Var[i] = 0;
  };
  SVMINT pos;
  svm_attrib the_att;
  for(pos=0;pos<capacity;pos++){
    for(i=0;i<the_set[pos].length;i++){
      the_att = (the_set[pos].example)[i];
      Exp[the_att.index] += the_att.att;
      Var[the_att.index] += the_att.att*the_att.att;
    };
    Exp[dim] += the_set[pos].y;
    Var[dim] += the_set[pos].y*the_set[pos].y;
  };
  for(i=0;i<=dim;i++){
    Exp[i] /= examples_total;
    Var[i] = (Var[i]-examples_total*Exp[i]*Exp[i])/(examples_total-1);
    if(Var[i] > 0){
      Var[i] = sqrt(Var[i]);
    }
    else{
      // numerical error
      Var[i] = 0;
    };
  };
  if(! scale_y){
    Exp[dim] = 0;
    Var[dim] = 0;
  };

  do_scale();

};


void example_set_c::do_scale(){
  // scale
  // precondition: Exp and Var are set.
  SVMINT i;
  SVMINT j=0;
  SVMINT k;
  //  SVMINT length;
  SVMINT nonzero=0;
  SVMINT pos;
  for(i=0;i<dim;i++){
    if(Var[i] != 0) nonzero++;
  };
  for(pos=0;pos<capacity;pos++){
    //    length = the_set[pos].length;

    // put zeros into vector, they might be scaled, kick constant atts out
    svm_attrib* new_example = new svm_attrib[nonzero];
    j = 0; // index in new vector
    k = 0; // index in old vector
    i=0;
    while((i<dim) && (j < nonzero)){
      if((k < the_set[pos].length) && (((the_set[pos].example)[k]).index < i)){
	k++;
      };
      if(Var[i] != 0){
	new_example[j].index = i;
	if(((the_set[pos].example)[k]).index == i){
	  new_example[j].att = ((the_set[pos].example)[k]).att;
	}
	else{
	  new_example[j].att = 0;
	};
	j++;
      };
      i++;
    };

    //    length = nonzero;
    the_set[pos].length = nonzero;
    delete []the_set[pos].example;
    the_set[pos].example = new_example;

    for(i=0;i<the_set[pos].length;i++){
      j = ((the_set[pos].example)[i]).index;
      if(0 != Var[j]){
	((the_set[pos].example)[i]).att = (((the_set[pos].example)[i]).att - Exp[j])/Var[j];
      }
      else{
	// shouldn't happen!
	((the_set[pos].example)[i]).att = 0; //  x - Exp = 0
      };
    };
    if(0 != Var[dim]){
      the_set[pos].y = (the_set[pos].y-Exp[dim])/Var[dim];
    }
    else{
      the_set[pos].y -= Exp[dim]; // don't know if to scale ys, so Exp could be 0 or y
    };

  };

  has_scale = 1;
};


void example_set_c::put_Exp_Var(SVMFLOAT *newExp, SVMFLOAT* newVar){
  // precondition: dim is ok

  if((newExp == 0) || (newVar == 0)){ return; };

  if(Exp==0) Exp = new SVMFLOAT[dim+1];
  if(Var==0) Var = new SVMFLOAT[dim+1];

  SVMINT i;

  for(i=0;i<=dim;i++){
    Exp[i] = newExp[i];
    Var[i] = newVar[i];
  };
};


void example_set_c::scale(SVMFLOAT *theconst, SVMFLOAT *thefactor,SVMINT scaledim){
  if((theconst == 0) || (thefactor == 0)) return;

  if(scaledim>dim) set_dim(scaledim);

  if(Exp==0) Exp = new SVMFLOAT[dim+1];
  if(Var==0) Var = new SVMFLOAT[dim+1];

  SVMINT i;

  for(i=0;i<scaledim;i++){
    Exp[i] = theconst[i];
    Var[i] = thefactor[i];
  };
  for(i=scaledim;i<dim;i++){
    Exp[i] = 0;
    Var[i] = 0;
  };
  Exp[dim] = theconst[scaledim];
  Var[dim] = thefactor[scaledim];

  do_scale();
};


SVMFLOAT example_set_c::unscale_y(const SVMFLOAT scaled_y){
  if((0 == Exp) || (0 == Var)){
    return scaled_y;
  }
  else if(0 == Var[dim]){
    return scaled_y+Exp[dim];
  }
  else{
    return (scaled_y*Var[dim]+Exp[dim]);
  };
};


void example_set_c::permute(){
  // permute the examples
  //  srand((unsigned int)time(0));
  svm_example dummy;
  SVMINT swap_pos;

  SVMINT pos;
  for(pos=0;pos<capacity-1;pos++){
    swap_pos = (SVMINT)((SVMFLOAT)(pos+1)*rand()/(RAND_MAX+1.0));
    dummy = the_set[swap_pos];
    the_set[swap_pos] = the_set[pos];
    the_set[pos] = dummy;
  };
  SVMINT i;
  SVMINT the_size;
  if(all_alphas != 0){
    the_size = size();
    for(i=0;i<the_size;i++){
      all_alphas[i] = get_alpha(i);
    };
  };
  if(all_ys != 0){
    the_size = size();
    for(i=0;i<the_size;i++){
      all_ys[i] = get_y(i);
    };
  };
};


void example_set_c::clear_alpha(){
  SVMINT i;
  for(i=0;i<capacity;i++){
    put_alpha(i,0);
  };
  if(all_alphas){
    for(i=0;i<capacity;i++){
      all_alphas[i] = 0;
    };
  };
};


SVMFLOAT example_set_c::sum(){
  // set examples in a consistent state.

  SVMFLOAT sum_alpha=0;
  SVMINT i;
  for(i=0;i<capacity;i++){
    sum_alpha += get_alpha(i);
  };
  return(sum_alpha);
};


void example_set_c::output_ys(std::ostream& data_stream) const{
  data_stream<<"# examples ys"<<std::endl;
  SVMINT i;
  for(i=0;i<examples_total;i++){
    data_stream<<(the_set[i].y)<<std::endl;
  };
};


void readnext(std::istream& i, char* s, const char delimiter){
  SVMINT pos=0;
  char next = i.peek();
  if(next == EOF){
    // set stream to eof
    next = i.get();
  };
  // skip whitespace
  while((! i.eof()) &&
	(('\n' == next) ||
	 (' ' == next)  ||
	 ('\t' == next) ||
	 ('\r' == next) ||
	 ('\f' == next))){
    i.get();
    next = i.peek();
    if(next == EOF){
      // set stream to eof
      next = i.get();
    };
  };
  // read next token
  if(delimiter == next){
    s[pos] = '0';
    pos++;
    next = i.peek();
    if(next == EOF){
      // set stream to eof
      next = i.get();
    };
  }
  else{
    while((! i.eof()) &&
	  ('\n' != next) &&
	  (' ' != next) &&
	  ('\t' != next) &&
	  ('\r' != next) &&
	  ('\f' != next) &&
	  (delimiter != next) &&
	  (pos < MAXCHAR-1)){
      s[pos] = i.get();
      pos++;
      next = i.peek();
      if(next == EOF){
	// set stream to eof
	next = i.get();
      };
    };
  };
  s[pos] = '\0';
  if(! (i.eof() || ('\n' == next))){
    // remove delimiter
    i.get();
  };
};


std::istream& operator>> (std::istream& data_stream, example_set_c& examples){
  // lower case, scale (y/n)
  char* s = new char[MAXCHAR]; // next item in the stream
  char* s2 = new char[MAXCHAR];
  long count=0; // number of examples read (does not necessarily equal examples_total
  char next=0; // first character in the stream
  char delimiter = examples.my_format.delimiter; // By which character are the numbers separated?
  int sparse = examples.my_format.sparse;
  int where_x = examples.my_format.where_x;  // format of the file
  int where_y = examples.my_format.where_y;
  int where_alpha = examples.my_format.where_alpha;
  SVMINT i,j;
  SVMINT given_total = 0; // what does the user say is the total of examples?
  SVMINT pos; // dummy for pos of attribute in example
  SVMINT dim = examples.get_dim();
  SVMFLOAT* new_example = new SVMFLOAT[dim+2]; // examples to be inserted

  while((next != EOF) && ('@' != next) && (! data_stream.eof())){
    try{
      next = data_stream.peek();
      if(next == EOF){
	// set stream to eof
	next = data_stream.get();
      };
      if(('@' == next) || (data_stream.eof())){
	// end of this section
      }
      else if(('\n' == next) ||
	      (' ' == next) ||
	      ('\r' == next) ||
	      ('\f' == next) ||
	      ('\t' == next)){
	// ignore
	next = data_stream.get();
      }
      else if('#' == next){
	// line contains commentary
	data_stream.getline(s,MAXCHAR);
      }
      else if(('+' == next) || ('-' == next) ||
	      ('y' == next) || ('a' == next) ||
	      ((next >= '0') && (next <= '9'))){
	// read an example
	pos = 0;
	new_example[dim] = 0;
	new_example[dim+1] = 0;
	if(sparse){
	  for(pos=0;pos<dim;pos++){
	    new_example[pos] = 0;
	  };
	  while((! data_stream.eof()) && ('\n' != data_stream.peek())){
	    readnext(data_stream,s,delimiter);
	    SVMINT spos = 0;
	    while((s[spos] != '\0') && (s[spos] != ':')){
	      spos++;
	    };
	    if(s[spos] == '\0'){
	      // read y
	      try{
		new_example[dim] = string2svmfloat(s);
	      }
	      catch(...){
		throw read_exception("Class is no number - could not read example");
	      };
	      examples.set_initialised_y();
	    }
	    else{
	      if(s[spos-1] == 'a'){
		// read alpha
		strncpy(s2,s+spos+1,MAXCHAR-spos);
		try{
		  new_example[dim+1] = string2svmfloat(s2);
		}
		catch(...){
		  throw read_exception("Alpha is no number - could not read example");
		};
		examples.set_initialised_alpha();
	      }
	      else if(s[spos-1] == 'y'){
		// read y
		strncpy(s2,s+spos+1,MAXCHAR-spos);
		try{
		  new_example[dim] = string2svmfloat(s2);
		}
		catch(...){
		  throw read_exception("Class is no number - could not read example");
		};
		examples.set_initialised_y();
	      }
	      else{
		// input index runs from 1 to dim (svmlight-compatibility):
		pos = atoi(s);
		if(pos <= 0){
		  throw read_exception("Index number not positive.");
		};
		if(pos>dim){
		  // raise dimension
		  examples.set_dim(pos);
		  SVMFLOAT* example_dummy = new SVMFLOAT[pos+2];
		  example_dummy[pos] = new_example[dim];
		  example_dummy[pos+1] = new_example[dim+1];
		  for(i=0;i<dim;i++){
		    example_dummy[i] = new_example[i];
		  };
		  for(i=dim;i<pos;i++){
		    example_dummy[i] = 0;
		  };
		  dim = pos;
		  delete []new_example;
		  new_example = example_dummy;
		};
		try{
		  new_example[pos-1] = string2svmfloat(s+spos+1);
		}
		catch(...){
		  char* t = new char[MAXCHAR];
		  strcpy(t,"Attribute is no number - could not read example: ");
		  t = strcat(t,s);
		  throw read_exception(t);
		};
	      };
	    };
	    while((! data_stream.eof()) &&
		  ((' ' == data_stream.peek()) ||
		   ('\t' == data_stream.peek()))){
	      data_stream.get();
	    };
	  };
	  pos = dim; // mark as ok
	}
	else{
	  // not sparse
	  for(int i=1;i<=3;i++) {
	    if(i ==  where_x){
	      // read attributes
	      if(dim <= 0) {
		// read & get dim
		char next_ws = data_stream.peek();
		if(next_ws == EOF){
		  // set stream to eof
		  next_ws = data_stream.get();
		};
		dim=0;
		pos = 0;
		while(!(data_stream.eof() || ('\n' == next_ws))){
		  // try to read another attribute
		  while((! data_stream.eof()) &&
			((' ' == next_ws) ||
			 ('\t' == next_ws))){
		    data_stream.get();
		    next_ws = data_stream.peek();
		    if(next_ws == EOF){
		      // set stream to eof
		      next_ws = data_stream.get();
		    };
		  };
		  if(!(data_stream.eof() || ('\n' == next_ws))){
		    // attribute is there, read it
		    if(pos == dim){
		      // double dim
		      dim = 2*dim+1;
		      SVMFLOAT* dummy = new_example;
		      new_example = new SVMFLOAT[dim+2];
		      new_example[dim] = dummy[pos];
		      new_example[dim+1] = dummy[pos+1];
		      for(j=0;j<pos;j++){
			new_example[j] = dummy[j];
		      };
		      delete []dummy;
		    };
		    // read example into act_pos
		    readnext(data_stream,s,delimiter);
		    try{
		      new_example[pos]= string2svmfloat(s);
		    }
		    catch(...){
		      throw read_exception("Attribute is no number - could not read example");
		    };
		    pos++;
		    next_ws = data_stream.peek();
		    if(next_ws == EOF){
		      // set stream to eof
		      next_ws = data_stream.get();
		    };
		  };
		};
		// line finished, set dim and exit
		if(where_y > where_x){
		  pos--;
		  // y at pos or pos+1 (one of xya xay xy)
		  if(where_y < where_alpha){
		    // xya
		    pos--;
		    new_example[dim] = new_example[pos];
		    new_example[dim+1] = new_example[pos+1];
		  }
		  else if(where_alpha < where_x){
		    // xy
		    new_example[dim] = new_example[pos];
		  }
		  else{
		    // xay
		    pos--;
		    SVMFLOAT dummy = new_example[pos]; // if pos==dim
		    new_example[dim] = new_example[pos+1];
		    new_example[dim+1] = dummy;
		  };
		}
		else if(where_alpha > where_x){
		  // xa
		  pos--;
		  new_example[dim+1] = new_example[pos];
		};
		SVMFLOAT* dummy = new_example;
		new_example = new SVMFLOAT[pos+2];
		for(j=0;j<pos;j++){
		  new_example[j] = dummy[j];
		};
		new_example[pos] = dummy[dim];
		new_example[pos+1] = dummy[dim+1];
		delete []dummy;
		dim = pos;
		examples.set_dim(dim);
		i=4;
	      }
	      else{
		// read dense data line
		for(pos=0;pos<dim;pos++){
		  readnext(data_stream,s,delimiter);
		  if(s[0] == '\0'){
		    throw read_exception("Not enough attributes - could not read examples");
		  };
		  try{
		    new_example[pos] = string2svmfloat(s);
		  }
		  catch(...){
		    char* t = new char[MAXCHAR];
		    strcpy(t,"Attribute is no number - could not read example: ");
		    t = strcat(t,s);
		    throw read_exception(t);
		  };
		};
	      };
	    }
	    else if(i == where_y){
	      // read classification
	      readnext(data_stream,s,delimiter);
	      if(s[0] == '\0'){
		throw read_exception("Not enough attributes - could not read examples");
	      };
	      try{
		new_example[dim] = string2svmfloat(s);
	      }
	      catch(...){
		throw read_exception("Class is no number - could not read example");
	      };
	      examples.set_initialised_y();
	    }
	    else if(i == where_alpha){
	      // read alpha
	      readnext(data_stream,s,delimiter);
	      if(s[0] == '\0'){
		throw read_exception("Not enough attributes - could not read examples");
	      };
	      try{
		new_example[dim+1] = string2svmfloat(s);
	      }
	      catch(...){
		throw read_exception("Alpha is no number - could not read example");
	      };
	      examples.set_initialised_alpha();
	    };
	  };
	};
	// insert examples, if ok.
	if(pos==dim){
	  // example ok, insert
	  examples.put_example(new_example);
	  count++;
	};
      }
      else{
	// line contains parameters
	data_stream >> s;
	if((0 == strcmp("dimension",s)) || (0==strcmp("dim",s))){
	  // dimension already set => error
	  SVMINT new_dim;
	  data_stream >> new_dim;
	  examples.set_dim(new_dim);
	  dim = new_dim;
	  if(new_example != 0){ delete []new_example; };
	  new_example = new SVMFLOAT[dim+2];
	}
	else if(0 == strcmp("number",s)){
	  // number of examples, check later for consistency
	  data_stream >> given_total;
	  if(given_total > 0){
	    // (examples.the_set).reserve((examples.the_set).size() + given_total);
	    examples.resize(examples.size()+given_total);
	  };
	}
	else if(0==strcmp("b",s)){
	  // hyperplane constant
	  data_stream >> s;
	  examples.b = string2svmfloat(s);
	}
	else if(0==strcmp("delimiter",s)){
	  data_stream >> s;
	  if((s[0] != '\0') && (s[1] != '\0')){
	    delimiter = s[1];
	  }
	  else if ((s[1] == '\0') && (s[0] != '\0')){
	    delimiter = s[0];
	    if(' ' == data_stream.peek()){
	      // if delimiter = ' ' we have only read one '
	      data_stream.get();
	      if(delimiter == data_stream.peek()){
		data_stream.get();
		delimiter = ' ';
	      };
	    };
	  }
	  else{
	    delimiter = ' ';
	  };
	  examples.my_format.delimiter = delimiter;
	}
	else if(0==strcmp("format",s)){
	  data_stream >> s;
	  if(0==strcmp("sparse",s)){
	    sparse = 1;
	  }
	  else{
	    sparse = 0;
	    where_x = 0;
	    where_y = 0;
	    where_alpha = 0;
	    for(int i=0;s[i] != '\0';i++){
	      if('x' == s[i]){
		where_x = i+1;
	      }
	      else if('y' == s[i]){
		where_y = i+1;
	      }
	      else if('a' == s[i]){
		where_alpha = i+1;
	      }
	      else{
		throw read_exception("Invalid format for examples");
	      };
	    };
	    if(0 == where_x){
	      throw read_exception("Invalid format for examples: x must be given");
	    };
	  };
	  if(0 == where_y){ examples.has_y = 0; };
	  if(0 == where_alpha){ examples.has_alphas = 0; };
	  examples.my_format.sparse = sparse;
	  examples.my_format.where_x = where_x;
	  examples.my_format.where_y = where_y;
	  examples.my_format.where_alpha = where_alpha;
	}
	else{
	  char* t = new char[MAXCHAR];
	  strcpy(t,"Unknown parameter: ");
	  strcat(t,s);
	  throw read_exception(t);
	};
      };
    }
    catch(general_exception g){
      // re-throw own exceptions
      if(new_example) delete []new_example;
      throw g;
    }
    catch(...){
      if(new_example) delete []new_example;
      throw read_exception("Error while reading from stream");
    };
  };
  if(new_example) delete []new_example;
  examples.compress();

  // check for consistency
  if((0 < given_total) && (count != given_total)){
    std::cout<<"WARNING: Wrong number of examples read ("<<count<<" read instead of "<<given_total<<")."<<std::endl;
  };
  delete []s;
  delete []s2;
  return data_stream;
};



std::ostream& operator<< (std::ostream& data_stream, example_set_c& examples){
  // output examples
  data_stream << "# svm example set" << std::endl;
  data_stream << "dimension "<< examples.dim << std::endl;
  data_stream << "number "<< examples.examples_total << std::endl;
  data_stream << "b " << examples.b << std::endl;
  char delimiter = examples.my_format.delimiter;
  if(delimiter != ' '){
    data_stream<<"delimiter '"<<delimiter<<"'"<<std::endl;
  };
  SVMINT total = examples.examples_total;
  SVMINT dim = examples.dim;
  SVMINT i;
  SVMINT pos;
  SVMINT j=0;
  svm_example the_example;
  // output examples;
  if(examples.my_format.sparse){
    data_stream<<"format "<<examples.my_format<<std::endl;
    for(i=0;i<total;i++){
      // output example i
      the_example = examples.get_example(i);
      if((examples.Exp != 0) && (examples.Var != 0)){
	for(pos=0;pos<the_example.length-1;pos++){
	  // output x_j
	  j = the_example.example[pos].index;
	  data_stream<<(the_example.example[pos].index+1)<<":";
	  if(0 != examples.Var[j]){
	    data_stream<<(the_example.example[pos].att*examples.Var[j]+examples.Exp[j]);
	  }
	  else{
	    data_stream<<the_example.example[pos].att+examples.Exp[j];
	  };
	  data_stream<<delimiter;
	}
	data_stream<<(the_example.example[the_example.length-1].index+1)<<":";
	if(0 != examples.Var[dim-1]){
	    data_stream<<(the_example.example[the_example.length-1].att*examples.Var[dim-1]+examples.Exp[dim-1]);
	}
	else{
	  data_stream<<the_example.example[the_example.length-1].att+examples.Exp[dim-1];
	};
	if(examples.has_y){
	  if(0 != examples.Var[dim]){
	    data_stream << delimiter << "y:" << examples.get_y(i)*examples.Var[dim]+examples.Exp[dim];
	  }
	  else{
	    data_stream << delimiter << "y:" << examples.get_y(i)+examples.Exp[dim];
	  };
	};
      }
      else{
	for(pos=0;pos<the_example.length-1;pos++){
	  data_stream<<(the_example.example[pos].index+1)<<":"
		     <<(the_example.example[pos].att)<<delimiter;
	};
	data_stream<<(the_example.example[the_example.length-1].index+1)<<":"
		   <<(the_example.example[the_example.length-1].att);
	if(examples.has_y){
	  data_stream << delimiter << "y:" << examples.get_y(i);
	};
      };
      if(examples.has_alphas){
	if(examples.get_alpha(i) != 0){
	  data_stream << delimiter << "a:" << examples.get_alpha(i);
	};
      };
      data_stream << std::endl;
    };
  }
  else{
    // output dense format
    int where_x = examples.my_format.where_x;
    int where_y = examples.my_format.where_y;
    int where_alpha = examples.my_format.where_alpha;

    // output computed values as well
    if((0 == where_y) && (examples.initialised_y())){
      examples.my_format.where_y = 4;
      where_y = 4;
    }
    if((0 == where_alpha) && (examples.initialised_alpha())){
      examples.my_format.where_alpha = 5;
      where_alpha = 5;
    }
    data_stream<<"format "<<examples.my_format<<std::endl;
    SVMINT pos;
    for(i=0;i<total;i++){
      // output example i
      the_example = examples.get_example(i);
      for(int s=1;s<=5;s++){
	if(where_x == s){
	  if(1 != s) data_stream<<delimiter;
	  pos=0; // index in example (0..the_example.length-1
	  for(j=0;j<dim;j++){
	    // output attribute j
	    if(j != 0) data_stream<<delimiter;
	    if((pos<the_example.length) && (the_example.example[pos].index == j)){
	      // output the_example.example[pos].att
	      if((examples.Exp != 0) && (examples.Var != 0)){
		if(0 != examples.Var[j]){
		  data_stream<<(the_example.example[pos].att*examples.Var[j]+examples.Exp[j]);
		}
		else{
		  data_stream<<the_example.example[pos].att+examples.Exp[j];
		};
	      }
	      else{
		data_stream<<the_example.example[pos].att;
	      };
	      if(pos<the_example.length-1) pos++;
	    }
	    else{
	      data_stream<<"0";
	    };
	  };
	}
	else if(where_y == s){
	  if(1 != s) data_stream<<delimiter;
	  if((examples.Exp != 0) && (examples.Var != 0)){
	    if(0 != examples.Var[dim]){
	      data_stream<<examples.get_y(i)*examples.Var[dim]+examples.Exp[dim];
	    }
	    else{
	      data_stream<<examples.get_y(i)+examples.Exp[dim];
	    };
	  }
	  else{
	    data_stream<<examples.get_y(i);
	  };
	}
	else if (where_alpha == s){
	  if(1 != s) data_stream<<delimiter;
	  data_stream<<examples.get_alpha(i);
	};
      };
      data_stream<<std::endl;
    };
  };
  return data_stream;
};
