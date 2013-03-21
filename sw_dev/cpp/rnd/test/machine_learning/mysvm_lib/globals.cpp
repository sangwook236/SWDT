#include "globals.h"


SVMFLOAT x_i(const svm_example x, const SVMINT i){
  // return x[i] by binary search
  SVMINT low=0;
  SVMINT high=x.length;
  SVMINT med;
  SVMFLOAT result;
  while(low<high){
    med = (low+high)/2;
    if(x.example[med].index>=i){
      high=med;
    }
    else{
      low=med+1;
    };
  };
  if((high < x.length) && (x.example[high].index==i)){
    result = x.example[high].att;
  }
  else{
    result = 0;
  };
  return result;
};


SVMFLOAT string2svmfloat(char* s){
  // number =~ [+-]?\d+([.]\d+)?([Ee][+-]?\d+)?
  int size = 0;
  while(s[size] != '\0') size++;

  int char_read=0;
  SVMFLOAT number=0;
  int sign = 1;
  // sign
  if((size > 0) && ('+' == s[0])){
    char_read++;
  }
  else if((size > 0) && ('-' ==s[0])){
    char_read++;
    sign = -1;
  };
  // digits before "."
  while((char_read<size) && (s[char_read] >= '0') && (s[char_read] <= '9')){
    number=number*10+(s[char_read]-'0');
    char_read++;
  };
  // digits after "."
  if((char_read<size) && (('.' == s[char_read]) || (',' == s[char_read]))){
    SVMFLOAT factor = 0.1;
    char_read++;
    while((char_read<size) && (s[char_read] >= '0') && (s[char_read] <= '9')){
      number=number+factor*(s[char_read]-'0');
      char_read++;
      factor *= 0.1;
    };    
  };
  if(sign<0){
    number = -number;
  };
  // exponent
  if((char_read<size) && (('e' == s[char_read]) || ('E' == s[char_read]))){
    sign = 1;
    char_read++;
    if((char_read<size) && ('+' == s[char_read])){
      char_read++;
    }
    else if((char_read<size) && ('-' == s[char_read])){
      char_read++;
      sign = -1;
    };
    int exponent=0;
    while((char_read<size) && (s[char_read] >= '0') && (s[char_read] <= '9')){
      exponent = exponent*10+(s[char_read]-'0');
      char_read++;
    };
    number = number*pow(10.0,sign*exponent);
  };
  if(char_read<size){
    throw no_number_exception();
  };
  return number;
};

long get_time(){
#ifdef use_time
  struct tms the_time;
  times(&the_time);
  return(the_time.tms_utime);
#else
  return 0;
#endif
};

general_exception::general_exception(char* the_error){ error_msg = the_error; }
general_exception::general_exception(){ error_msg = ""; }

read_exception::read_exception(char* the_error){ error_msg = the_error; }
read_exception::read_exception(){ error_msg = ""; }


std::ostream& operator<< (std::ostream& data_stream, example_format& format){
  if(format.sparse){
    data_stream<<"sparse";
  }
  else{
    for(int i=1;i<=5;i++){
      if(format.where_x == i){
	data_stream<<"x";
      }
      else if(format.where_y == i){
	data_stream<<"y";
      }
      else if (format.where_alpha == i){
	data_stream<<"a";
      };
    };
  };
  return data_stream;
};
