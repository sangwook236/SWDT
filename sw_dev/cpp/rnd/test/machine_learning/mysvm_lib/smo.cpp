#include "smo.h"


smo_c::smo_c(){
  n=0;
  sum=0;
  x=0;
  qp=0;
  lambda_eq=0;
  is_zero = 1e-10;
  max_allowed_error= 1e-3;
  max_iteration = 10000;
};


smo_c::smo_c(const SVMFLOAT new_is_zero, const SVMFLOAT new_max_allowed_error, const SVMINT new_max_iteration){
  n=0;
  sum=0;
  x=0;
  qp=0;
  lambda_eq=0;
  init(new_is_zero,new_max_allowed_error,new_max_iteration);
};


void smo_c::init(const SVMFLOAT new_is_zero, const SVMFLOAT new_max_allowed_error, const SVMINT new_max_iteration){
  is_zero = new_is_zero;
  max_allowed_error = new_max_allowed_error;
  max_iteration = new_max_iteration;
};


void smo_c::set_max_allowed_error(SVMFLOAT new_max_allowed_error){
  if(new_max_allowed_error>0){
    max_allowed_error = new_max_allowed_error;
  };
};


inline
SVMFLOAT smo_c::x2tox1(const SVMFLOAT x2, const int id, 
		       const SVMFLOAT A1, const SVMFLOAT b){
  SVMFLOAT x1;
  if(id){
    x1 = -x2;
  }
  else{
    x1 = x2;
  };
  if(A1>0){
    x1+=b;
  }
  else{
    x1 -= b;
  };
  return x1;
};


inline
SVMFLOAT smo_c::x1tox2(const SVMFLOAT x1, const int id, 
		       const SVMFLOAT A2, const SVMFLOAT b){
  SVMFLOAT x2;
  if(id){
    x2 = -x1;
  }
  else{
    x2 = x1;
  };
  if(A2>0){
    x2+=b;
  }
  else{
    x2 -= b;
  };
  return x2;
};


inline
void smo_c::simple_solve(SVMFLOAT* x1, SVMFLOAT* x2,
			 const SVMFLOAT H1, const SVMFLOAT H2,
			 const SVMFLOAT c0, 
			 const SVMFLOAT c1, const SVMFLOAT c2,
			 const SVMFLOAT A1, const SVMFLOAT A2,
			 const SVMFLOAT l1, const SVMFLOAT l2,
			 const SVMFLOAT u1, const SVMFLOAT u2){
  /*
   * H1*x1^2+H2*x2^2+c0*x1*x2+c1*x1+c2*x2 -> min
   *
   * w.r.t.: A1*x1+A2*x2=const
   *         l1 <= x1 <= u1
   *         l2 <= x2 <= u2
   *
   */

  SVMFLOAT t;

  SVMFLOAT den;
  den = H1+H2;
  if(((A1 > 0) && (A2 > 0)) ||
     ((A1 < 0) && (A2 < 0))){
    den -= c0;
  }
  else{
    den += c0;
  };
  den*=2;
  if(den != 0){
    SVMFLOAT num;
    num = -2*H1*(*x1)-(*x2)*c0-c1;
    if(A1<0){
      num = -num;
    };
    if(A2>0){
      num += 2*H2*(*x2)+(*x1)*c0+c2;
    }
    else{
      num -= 2*H2*(*x2)+(*x1)*c0+c2;
    };

    t = num/den;
    
    SVMFLOAT up;
    SVMFLOAT lo;
    if(A1>0){
      lo = l1-(*x1);
      up = u1-(*x1);
    }
    else{
      lo = (*x1)-u1;
      up = (*x1)-l1;
    };
    if(A2<0){
      if(l2-(*x2) > lo) lo = l2-(*x2);
      if(u2-(*x2) < up) up = u2-(*x2);
    }
    else{
      if((*x2)-l2 < up) up =(*x2)-l2;
      if((*x2)-u2 > lo) lo = (*x2)-u2;
    };
    
    if(t < lo){
      t = lo;
    };
    if(t > up){
      t = up;  
    };
  }
  else{
    // den = 0 => linear target function => set x at bound
    SVMFLOAT factor;
    factor = 2*H1*(*x1)+(*x2)*c0+c1;
    if(A1<0){
      factor = -factor;
    };
    if(A2>0){
      factor -= 2*H2*(*x2)+(*x1)*c0+c2;
    }
    else{
      factor += 2*H2*(*x2)+(*x1)*c0+c2;
    };
    if(factor>0){
      // t = lo
      if(A1>0){
	t = l1-(*x1);
      }
      else{
	t = (*x1)-u1;
      };
      if(A2<0){
	if(l2-(*x2) > t) t = l2-(*x2);
      }
      else{
	if((*x2)-u2 > t) t = (*x2)-u2;
      };
    }
    else{
      // t = up
      if(A1>0){
	t = u1-(*x1);
      }
      else{
	t = (*x1)-l1;
      };
      if(A2<0){
	if(u2-(*x2) < t) t = u2-(*x2);
      }
      else{
	if((*x2)-l2 < t) t =(*x2)-l2;
      };
    };
  };

  // calc new x from t
  if(A1>0){
    (*x1) += t;
  }
  else{
    (*x1) -= t;
  };
  if(A2>0){
    (*x2) -= t;
  }
  else{
    (*x2) += t;
  };

  if(*x1-l1 <= is_zero){
    *x1 = l1; 
  }
  else if(*x1-u1 >= -is_zero){
    *x1 = u1;
  };
  if(*x2-l2 <= is_zero){
    *x2 = l2; 
  }
  else if(*x2-u2 >= -is_zero){
    *x2 = u2;
  };

};


int smo_c::minimize_ij(const SVMINT i, const SVMINT j){
  // minimize xi, xi with simple_solve

  SVMFLOAT sum_i; // sum_k Hik x_k
  SVMFLOAT sum_j;

  // init sum_i,j
  sum_i=sum[i];
  sum_j=sum[j];
  sum_i -= qp->H[i*(n+1)]*x[i];
  sum_i -= qp->H[i*n+j]*x[j];
  sum_j -= qp->H[j*n+i]*x[i];
  sum_j -= qp->H[j*(n+1)]*x[j];

  SVMFLOAT old_xi = x[i];
  SVMFLOAT old_xj = x[j];

  simple_solve(&(x[i]), &(x[j]),
	       qp->H[i*(n+1)]/2, qp->H[j*(n+1)]/2,
	       qp->H[i*n+j],
	       sum_i, sum_j,
	       qp->A[i], qp->A[j],
	       qp->l[i], qp->l[j],
	       qp->u[i], qp->u[j]);

  SVMFLOAT target;
  target = (old_xi-x[i])*(qp->H[i*(n+1)]/2*(old_xi+x[i])+sum_i)
    +(old_xj-x[j])*(qp->H[j*(n+1)]/2*(old_xj+x[j])+sum_j)
    +qp->H[i*n+j]*(old_xi*old_xj-x[i]*x[j]);
  if(target < 0){
    //       cout<<"increase on SMO: "<<target<<endl;
    x[i] = old_xi;
    x[j] = old_xj;
    old_xi=0;
    old_xj=0;
  }
  else{
    old_xi-=x[i];
    old_xj-=x[j];
    if((old_xi != 0) || (old_xj != 0)){
      SVMINT k;
      SVMFLOAT* Hin = &(qp->H[i*n]);
      SVMFLOAT* Hjn = &(qp->H[j*n]);
      for(k=0;k<n;k++){
	sum[k]-=Hin[k]*old_xi;
	sum[k]-=Hjn[k]*old_xj;
      };
    };
  };

  return((abs(old_xi) > is_zero) || (abs(old_xj) > is_zero));
};


void smo_c::calc_lambda_eq(){
  SVMFLOAT lambda_eq_sum = 0;
  SVMINT count = 0;
  SVMINT i;
  for(i=0;i<qp->n;i++){
    if((x[i] > qp->l[i]) && (x[i]<qp->u[i])){
      if(qp->A[i]>0){
	lambda_eq_sum-= sum[i];
      }
      else{
	lambda_eq_sum+= sum[i];
      };
      count++;
    };
  };
  if(count>0){
    lambda_eq_sum /= (SVMFLOAT)count;
  }
  else{
    SVMFLOAT lambda_min = -infinity;
    SVMFLOAT lambda_max = infinity;
    SVMFLOAT nabla;
    for(i=0;i<qp->n;i++){
      nabla = sum[i];
      if(x[i] <= qp->l[i]){
	// lower bound
	if(qp->A[i]>0){
	  if(-nabla > lambda_min){
	    lambda_min = -nabla;
	  };
	}
	else{
	  if(nabla < lambda_max){
	    lambda_max = nabla;
	  };
	};
      }
      else{
	// upper bound
	if(qp->A[i]>0){
	  if(-nabla < lambda_max){
	    lambda_max = -nabla;
	  };
	}
	else{
	  if(nabla > lambda_min){
	    lambda_min = nabla;
	  };
	};
      };
    };
    if(lambda_min > -infinity){
      if(lambda_max < infinity){
	lambda_eq_sum = (lambda_max+lambda_min)/2;
      }
      else{
	lambda_eq_sum = lambda_min;
      };
    }
    else{
      lambda_eq_sum = lambda_max;
    };
  };
  lambda_eq = lambda_eq_sum;
};


void smo_c::set_qp(quadratic_program* the_qp){
  qp = the_qp;
  if(qp->n != n){
    n = qp->n;
    if(sum){
      delete []sum;
    };
    sum = new SVMFLOAT[n];
  };
};


int smo_c::smo_solve(quadratic_program* the_qp,SVMFLOAT* the_x){
  int error=0;

  x = the_x;
  set_qp(the_qp);

  SVMINT i;
  SVMINT j;
  for(i=0;i<n;i++){
    sum[i] = qp->c[i];
    SVMFLOAT* Hin = &(qp->H[i*n]);
    for(j=0;j<n;j++){
      sum[i] += Hin[j]*x[j];
    };
  };

  SVMINT iteration=0;
  SVMFLOAT this_error;
  SVMFLOAT this_lambda_eq;
  SVMFLOAT max_lambda_eq=0;
  SVMFLOAT max_error = -infinity;
  SVMFLOAT min_error = infinity;
  SVMINT max_i = 0;
  SVMINT min_i = 1;
  SVMINT old_max_i=-1;

  while(1){
    // get i with largest KKT error
    if(! error){
      //      cout<<"l";
      calc_lambda_eq();
      max_error = -infinity;
      min_error = infinity;
      max_i = 0;
      min_i = 1;
      // heuristic for i
      for(i=0;i<n;i++){
	if(x[i] <= qp->l[i]){
	  // at lower bound
	  this_error = -sum[i];
	  if(qp->A[i]>0){
	    this_lambda_eq = this_error;
	    this_error -= lambda_eq;
	  }
	  else{
	    this_lambda_eq = -this_error;
	    this_error += lambda_eq;
	  };
	}
	else if(x[i] >= qp->u[i]){
	  // at upper bound
	  this_error = sum[i];
	  if(qp->A[i]>0){
	    this_lambda_eq = -this_error;
	    this_error += lambda_eq;
	  }
	  else{
	    this_lambda_eq = this_error;
	    this_error -= lambda_eq;
	  };
	}
	else{
	  // between bounds
	  this_error = sum[i];
	  if(qp->A[i]>0){
	    this_lambda_eq = -this_error;
	    this_error += lambda_eq;
	  }
	  else{
	    this_lambda_eq = this_error;
	    this_error -= lambda_eq;
	  };
	  if(this_error<0) this_error = -this_error;
	}
	if((this_error>max_error) && (old_max_i != i)){
	  max_i = i;
	  max_error = this_error;
	  max_lambda_eq = this_lambda_eq;
	};
      };
      old_max_i = max_i;
    }
    else{
      // heuristic didn't work
      max_i = (max_i+1)%n;
    };

    // problem solved?
    if((max_error<=max_allowed_error) && (iteration>2)){
      error=0;
      break;
    };

    ////////////////////////////////////////////////////////////

    // new!!! find element with maximal diff to max_i
    // loop would be better
    SVMFLOAT max_diff = -1;
    SVMFLOAT this_diff;
    int n_up; // not at upper bound
    int n_lo;
    if(x[max_i] <= qp->l[max_i]){
      // at lower bound
      n_lo = 0;
    }
    else{
      n_lo = 1;
    };
    if(x[max_i] >= qp->u[max_i]){
      // at lower bound
      n_up=0;
    }
    else{
      n_up=1;
    };

    min_i = (max_i+1)%n;
    for(i=0;i<n;i++){
      if((i != max_i) &&
	 (n_up || (x[i] < qp->u[i])) &&
	 (n_lo || (x[i] > qp->l[i]))){
	if(x[i] <= qp->l[i]){
	  // at lower bound
	  this_error = -sum[i];
	  if(qp->A[i]<0){
	    this_error = -this_error;
	  };
	}
	else{
	  // between bounds
	  this_error = sum[i];
	  if(qp->A[i]>0){
	    this_error = -this_error;
	  };
	};
	this_diff = abs(this_error - max_lambda_eq);
	if(this_diff>max_diff){
	  max_diff = this_diff;
	  min_i = i;
	};
      };
    };


    ////////////////////////////////////////////////////////////

    // optimize
    SVMINT it=1;
    while((0 == minimize_ij(min_i,max_i)) && (it<n)){
      it++;
      min_i = (min_i+1)%n;
      if(min_i == max_i){
      	min_i = (min_i+1)%n;
      };
    };
    if(it==n){
      error=1;
    }
    else{
      error=0;
    };

    // time up?
    iteration++;
    if(iteration>max_iteration){
      calc_lambda_eq();
      error+=1;
      break;
    };
  };

  return error;
};


SVMFLOAT smo_c::get_lambda_eq(){
  return lambda_eq;
};


/**
 *
 * unbiased SVM
 *
 **/


int smo_c::minimize_i(const SVMINT i){
  // minimize xi with simple_solve

  SVMFLOAT sum_i; // sum_{k\ne i} Hik x_k +c[i]

  // init sum_i,j
  sum_i=sum[i];
  sum_i -= qp->H[i*(n+1)]*x[i];  

  SVMFLOAT old_xi = x[i];

  x[i] = -sum_i/(qp->H[i*(n+1)]);
  if(x[i] < qp->l[i]){
    x[i] = qp->l[i];
  }
  else if(x[i] > qp->u[i]){
    x[i] = qp->u[i];
  };

  int ok;

  SVMFLOAT target;

  target = (old_xi-x[i])*(qp->H[i*(n+1)]/2*(old_xi+x[i])+sum_i);
  if(target < 0){
    //       cout<<"increase on SMO: "<<target<<endl;
    x[i] = old_xi;
    old_xi=0;
    ok=0;
  }
  else{
    old_xi-=x[i];
    SVMINT k;
    for(k=0;k<n;k++){
      sum[k]-=qp->H[i*n+k]*old_xi;
    };
    ok=1;
  };

  if(abs(old_xi) > is_zero){
    ok =1;
  }
  else{
    ok=0;
  };
  return ok;
};


int smo_c::smo_solve_single(quadratic_program* the_qp,SVMFLOAT* the_x){
  int error=0;

  x = the_x;
  set_qp(the_qp);

  SVMINT i;
  SVMINT j;
  for(i=0;i<n;i++){
    sum[i] = qp->c[i];
    for(j=0;j<n;j++){
      sum[i] += qp->H[i*n+j]*x[j];
    };
  };

  SVMINT iteration=0;
  SVMFLOAT this_error;
  SVMFLOAT max_error = -infinity;
  SVMINT max_i = 0;
  SVMINT old_max_i=-1;

  lambda_eq = 0.0;

  while(1){
    // get i with largest KKT error
    if(! error){
      //      cout<<"l";
      max_error = -infinity;
      max_i = 0;
      // heuristic for i
      for(i=0;i<n;i++){
	if(x[i] <= qp->l[i]){
	  // at lower bound
	  this_error = -sum[i];
	}
	else if(x[i] >= qp->u[i]){
	  // at upper bound
	  this_error = sum[i];
	}
	else{
	  // between bounds
	  this_error = sum[i];
	  if(this_error<0) this_error = -this_error;
	}
	if((this_error>max_error) && (old_max_i != i)){
	  max_i = i;
	  max_error = this_error;
	};
      };
      old_max_i = max_i;
    }
    else{
      // heuristic didn't work
      max_i = (max_i+1)%n;
    };

    // problem solved?
    if((max_error<=max_allowed_error) && (iteration>2)){
      error=0;
      break;
    };

    ////////////////////////////////////////////////////////////

    // optimize
    SVMINT it=minimize_i(max_i);

    if(it != 0){
      error=1;
    }
    else{
      error=0;
    };

    // time up?
    iteration++;
    if(iteration>max_iteration){
      error+=1;
      break;
    };
  };

  return error;
};


/**
 *
 *  nuSVM
 *
 **/

SVMFLOAT smo_c::get_lambda_nu(){
  return lambda_nu;
};


void smo_c::calc_lambda_nu(){
  SVMFLOAT lambda_pos_sum = 0;
  SVMFLOAT lambda_neg_sum = 0;
  SVMINT countpos = 0;
  SVMINT countneg = 0;
  SVMINT i;
  for(i=0;i<qp->n;i++){
    if((x[i] > qp->l[i]) && (x[i]<qp->u[i])){
      if(qp->A[i]>0){
	lambda_pos_sum += sum[i];
	countpos++;
      }
      else{
	lambda_neg_sum += sum[i];
	countneg++;
      };
    };
  };
  if((countpos>0) && (countneg>0)){
    lambda_pos_sum /= (SVMFLOAT)countpos;
    lambda_neg_sum /= (SVMFLOAT)countneg;
    lambda_eq = -(lambda_pos_sum-lambda_neg_sum)/2;
    lambda_nu = -(lambda_pos_sum+lambda_neg_sum)/2;
  }
  else{
    if(countpos>0){
      lambda_eq = -lambda_pos_sum / (SVMFLOAT)countpos;
      lambda_eq /= 2;
      lambda_nu = lambda_eq;
    }
    else if(countneg>0){
      lambda_eq = -lambda_neg_sum / (SVMFLOAT)countneg;
      lambda_eq /= 2;
      lambda_nu = lambda_eq;
    }
    else{
      calc_lambda_eq();
      lambda_nu=0;
    };
  };
};


int smo_c::smo_solve_const_sum(quadratic_program* the_qp,SVMFLOAT* the_x){
  // solve optimization problem keeping sum x_i fixed
  int error=0;

  x = the_x;
  set_qp(the_qp);

  SVMFLOAT target=0;
  SVMINT i;
  SVMINT j;
  for(i=0;i<n;i++){
    sum[i] = 0;
    for(j=0;j<n;j++){
      sum[i] += qp->H[i*n+j]*x[j];
    };
    target += x[i]*sum[i]/2;
    target += qp->c[i]*x[i];
    sum[i] += qp->c[i];
  };

  SVMINT iteration=0;
  SVMFLOAT this_error;
  SVMFLOAT max_error = -infinity;
  SVMFLOAT min_error_pos = infinity;
  SVMFLOAT min_error_neg = infinity;
  SVMINT max_i = 0;
  SVMINT min_i = 1;
  SVMINT min_i_pos = 1;
  SVMINT min_i_neg = 1;
  SVMINT old_min_i=-1;
  SVMINT old_max_i=-1;
  int use_sign=1;
  while(1){
    // get i with largest KKT error
    if(! error){
      use_sign = -use_sign;
      calc_lambda_nu();
      max_error = -infinity;
      min_error_pos = infinity;
      min_error_neg = infinity;
      max_i = (old_max_i+1)%n;
      // heuristic for i
      for(i=0;i<n;i++){
	if(x[i] <= qp->l[i]){
	  // at lower bound
	  this_error = -sum[i]-lambda_nu;
	  if(qp->A[i]>0){
	    this_error -= lambda_eq;
	  }
	  else{
	    this_error += lambda_eq;
	  };
	}
	else if(x[i] >= qp->u[i]){
	  // at upper bound
	  this_error = sum[i]+lambda_nu;
	  if(qp->A[i]>0){
	    this_error += lambda_eq;
	  }
	  else{
	    this_error -= lambda_eq;
	  };
	}
	else{
	  // between bounds
	  this_error = sum[i]+lambda_nu;
	  if(qp->A[i]>0){
	    this_error += lambda_eq;
	  }
	  else{
	    this_error -= lambda_eq;
	  };
	  if(this_error<0) this_error = -this_error;
	}
	if(this_error>max_error){
	  if((old_max_i != i) && (qp->A[i] == use_sign)){
	    // look for specific sign
	    max_i = i;
	  };
	  max_error = this_error;
	};
	if((qp->A[i]>0) && (this_error<=min_error_pos) && (i != old_min_i)){
	  min_i_pos = i;
	  min_error_pos = this_error;
	};
	if((qp->A[i]<0) && (this_error<=min_error_neg) && (i != old_min_i)){
	  min_i_neg = i;
	  min_error_neg = this_error;
	};
      };

      old_max_i = max_i;
      // look for minimal error with same sign as max_i
      if(qp->A[max_i]>0){
	min_i = min_i_pos;
      }
      else{
	min_i = min_i_neg;
      };
      old_min_i = min_i;
    }
    else{
      // heuristic didn't work
      max_i = (max_i+1)%n;
      min_i = (max_i+1)%n;
    };

    // problem solved?
    if((max_error<=max_allowed_error) && (iteration > 2)){
      error=0;
      break;
    };

    // optimize
    SVMINT it=1; // n-1 iterations
    error=1;
    while((error) && (it<n)){
      if(qp->A[min_i] == qp->A[max_i]){
	error = ! minimize_ij(min_i,max_i);
      };
      it++;
      min_i = (min_i+1)%n;
      if(min_i == max_i){
      	min_i = (min_i+1)%n;
      };
    };
    // time up?
    iteration++;
    if(iteration>max_iteration){
      calc_lambda_nu();
      error+=1;
      break;
    };
  };

  SVMFLOAT ntarget=0;
  for(i=0;i<qp->n;i++){
    for(j=0;j<qp->n;j++){
      ntarget += x[i]*qp->H[i*qp->n+j]*x[j]/2;
    };
    ntarget += qp->c[i]*x[i];
  };

  if(target<ntarget){
    error++;
  };

  return error;
};
