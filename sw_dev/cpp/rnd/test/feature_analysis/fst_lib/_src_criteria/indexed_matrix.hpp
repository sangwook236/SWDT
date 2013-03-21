#ifndef FSTINDEXEDMATRIX_H
#define FSTINDEXEDMATRIX_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    indexed_matrix.hpp
   \brief   Matrix representation and operations, allows operation in selected subspace
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    October 2010
   \version 3.0.0.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/, 
                tested with version 3.00)
   \note    Note that LibSVM is required for SVM related tools only,
            as demonstrated in demo12t.cpp, demo23.cpp, demo25t.cpp, demo32t.cpp, etc.

*/ /* 
=========================================================================
Copyright:
  * FST3 software (with exception of any externally linked libraries) 
    is copyrighted by Institute of Information Theory and Automation (UTIA), 
    Academy of Sciences of the Czech Republic.
  * FST3 source codes as presented here do not contain code of third parties. 
    FST3 may need linkage to external libraries to exploit its functionality
    in full. For details on obtaining and possible usage restrictions 
    of external libraries follow their original sources (referenced from
    FST3 documentation wherever applicable).
  * FST3 software is available free of charge for non-commercial use. 
    Please address all inquires concerning possible commercial use 
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz).
  * Derivative works based on FST3 are permitted as long as they remain
    non-commercial only.
  * Re-distribution of FST3 software is not allowed without explicit
    consent of the copyright holder.
Disclaimer of Warranty:
  * FST3 software is presented "as is", without warranty of any kind, 
    either expressed or implied, including, but not limited to, the implied 
    warranties of merchantability and fitness for a particular purpose. 
    The entire risk as to the quality and performance of the program 
    is with you. Should the program prove defective, you assume the cost 
    of all necessary servicing, repair or correction.
Limitation of Liability:
  * The copyright holder will in no event be liable to you for damages, 
    including any general, special, incidental or consequential damages 
    arising out of the use or inability to use the code (including but not 
    limited to loss of data or data being rendered inaccurate or losses 
    sustained by you or third parties or a failure of the program to operate 
    with any other programs).
========================================================================== */

#include <boost/smart_ptr.hpp>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstring> // memcpy
#include "error.hpp"
#include "global.hpp"
#include "indexed_vector.hpp"

/*============== Template parameter type naming conventions ==============
--------- Numeric types: -------------------------------------------------
DATATYPE - data sample values - usually real numbers (but may be integers
          in text processing etc.)
REALTYPE - must be real numbers - for representing intermediate results of 
          calculations like mean, covariance etc.
IDXTYPE - index values for enumeration of data samples - (nonnegative) integers, 
          extent depends on numbers of samples in data
DIMTYPE - index values for enumeration of features (dimensions), or classes (not 
          class sizes) - (nonnegative) integers, usually lower extent than IDXTYPE, 
          but be aware of expressions like _classes*_features*_features ! 
          in linearized representations of feature matrices for all classes
BINTYPE - feature selection marker type - represents ca. <10 different feature 
          states (selected, deselected, sel./desel. temporarily 1st nested loop, 2nd...)
RETURNTYPE - criterion value: real value, but may be extended in future to support 
          multiple values 
--------- Class types: ---------------------------------------------------
SUBSET       - class of class type Subset 
CLASSIFIER   - class implementing interface defined in abstract class Classifier 
EVALUATOR    - class implementing interface defined in abstract class Sequential_Step 
DISTANCE     - class implementing interface defined in abstract class Distance 
DATAACCESSOR - class implementing interface defined in abstract class Data_Accessor 
INTERVALCONTAINER - class of class type TIntervaller 
CONTAINER    - STL container of class type TInterval  
========================================================================== */

namespace FST {

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Vector;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Matrix;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& mr);

//! Matrix representation and operations, allows operation in selected subspace
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Matrix {
public:
	typedef boost::shared_ptr<SUBSET> PSubset;
	Indexed_Matrix() : _n_max(0), _n(0), _sign(0), _d(0) {notify("Indexed_Matrix (empty) constructor");}
	Indexed_Matrix(const DIMTYPE n, bool LUstorage=false) : _n_max(0), _n(0), _sign(0), _d(0) {notify("Indexed_Matrix constructor"); reset(n,LUstorage);}
	Indexed_Matrix(const Indexed_Matrix& im); // copy-constructor
	Indexed_Matrix& operator=(Indexed_Matrix const &im);
	~Indexed_Matrix() {notify("Indexed_Matrix destructor");}
	void reset(const DIMTYPE n_max, bool LUstorage=false); //!< reallocates
	void redim(const DIMTYPE n=0); //!< keeps main allocations, just resets index arrays and adjusts _n; n==0 here equals _n_max
	
	DATATYPE& at_raw(const DIMTYPE row, const DIMTYPE col) {assert(row>=0 && row<_n); assert(col>=0 && col<_n); assert(_data); return _data[row*_n+col];}
	DATATYPE& at(const DIMTYPE row, const DIMTYPE col) {assert(row>=0 && row<_d); assert(col>=0 && col<_d); assert(_index); assert(_data); return _data[_index[row]*_n+_index[col]];}
	DATATYPE* get_row_raw(const DIMTYPE row) {assert(row>=0 && row<_n); assert(_data); return &(_data[row*_n]);}
	DATATYPE* get_row(const DIMTYPE row) {assert(row>=0 && row<_d); assert(_index); assert(_data); return &(_data[_index[row]*_n]);}

	void narrow_to(const PSubset sub); //!< fill _index[] -> access to data is then mapped to sub-matrix
	void denarrow(); //!< reset _index[] -> access is reset to the full matrix
	DIMTYPE get_n_max() const {return _n_max;}
	DIMTYPE get_n() const {return _n;}
	DIMTYPE get_d() const {return _d;}
	//void swap_rows(const IDXTYPE row1, const IDXTYPE row2);
	//void swap_cols(const IDXTYPE col1, const IDXTYPE col2);
	
	// "raw" operations, ignoring index[]
	void set_all_raw_to(const DATATYPE val);
	void add_raw(const Indexed_Matrix& mr);
	void add_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul);
	void copy_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul);
	void copy_raw(const Indexed_Matrix& mr);
	
	// "soft" methods - assumes source to be indexed sub-matrix, writes non-indexed (raw) result
	void LUdecompose(Indexed_Matrix &result); //!< returns matrix+vector+sign
	void solve_equations(Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &result_vec, const Indexed_Matrix &LU_matrix, Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &coef_vec); //!< returns vector
	void invert(Indexed_Matrix &result, const Indexed_Matrix &LU_matrix); //!< returns matrix
	DATATYPE determinant(const Indexed_Matrix &LU_matrix); //!< returns scalar
	
	friend std::ostream& operator<< <>(std::ostream& os, const Indexed_Matrix& mr);
protected:
	// NOTE: matrix size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max*_n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max. If Indexed_Matrix servers as output buffer
	//                to store matrices of various (reduced) dimensionalites, this is needed for correct functionality
	//                of at(), at_raw() etc. (both "raw" and "non-raw" matrix item access methods)
	//       _d     - "virtual subset" dimensionality, _d<=_n. Assuming the matrix holds _n*_n values (_n consecutively stored
	//                rows of _n values each), a virtual sub-matrix can be defined and accessed using at() ("non-raw" methods)
	DIMTYPE _n_max; //!< allocated space can hold _n_max*_n_max matrix
	DIMTYPE _n; //!< full feature set dimensionality
	boost::scoped_array<DATATYPE> _data; //!< raw matrix data
	// if used as LU decomposition target
	boost::scoped_array<DIMTYPE> _permut; //!< LU permutation index
	int _sign; //!< permutation sign (renamed obsolete d)

	// sub-matrix access tools
	DIMTYPE _d; //!< virtual sub-matrix (as represented by _index) dimensionality
	boost::scoped_array<DIMTYPE> _index; //!< maps virtual sub-matrix _d dimensions to original _n dimensions

	// temporary needed inside LUdecompose()
	boost::scoped_array<DATATYPE> _tmp_rowscale; //!< (lazy allocation)
	boost::scoped_array<DIMTYPE> _tmp_whereis; //!< (lazy allocation)
	// temporaries needed inside invert()
	boost::scoped_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _tmp_eq_coefs; //!< invert coefs (lazy allocation)
	boost::scoped_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _tmp_eq_roots; //!< to store/pass intermediate results in invert() etc. (lazy allocation)
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::Indexed_Matrix(const Indexed_Matrix& im) : // copy-constructor
	_n_max(im._n_max),
	_n(im._n),
	_sign(im._sign),
	_d(im._d)
{
	notify("Indexed_Matrix copy-constructor.");
	if(_n_max>0)
	{
		_data.reset(new DATATYPE[_n_max*_n_max]); memcpy((void *)_data.get(),(void *)(im._data).get(),sizeof(DATATYPE)*_n_max*_n_max);
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(im._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
		if(im._permut)       {_permut.reset(new DIMTYPE[_n_max]); memcpy((void *)_permut.get(),(void *)(im._permut).get(),sizeof(DIMTYPE)*_n_max);}
		if(im._tmp_whereis)  {_tmp_whereis.reset(new DIMTYPE[_n_max]); memcpy((void *)_tmp_whereis.get(),(void *)(im._tmp_whereis).get(),sizeof(DIMTYPE)*_n_max);}
		if(im._tmp_rowscale)	{_tmp_rowscale.reset(new DATATYPE[_n_max]); memcpy((void *)_tmp_rowscale.get(),(void *)(im._tmp_rowscale).get(),sizeof(DATATYPE)*_n_max);}
	}
	if(im._tmp_eq_roots) _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_roots))); // solve_equations result
	if(im._tmp_eq_coefs)	_tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_coefs))); // equations' right hand side coefficients
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::operator=(Indexed_Matrix const &im)
{
	notify("Indexed_Matrix operator=.");
	assert( (im._n_max>0 && im._data && im._index) || (im._n_max==0 && !im._data && !im._index) );
	assert( !im._permut || (im._permut && im._n_max > 0) );
	assert( !im._tmp_whereis || (im._tmp_whereis && im._n_max > 0) );
	assert( !im._tmp_rowscale || (im._tmp_rowscale && im._n_max > 0) );
	if (this != &im)
	{
		_n=im._n;
		_d=im._d;
		_sign=im._sign;

		if(im._n_max>0)
		{
			if(_n_max!=im._n_max)
			{
				_data.reset(new DATATYPE[im._n_max*im._n_max]);	
				_index.reset(new DIMTYPE[im._n_max]);
			}
			memcpy((void *)_data.get(),(void *)(im._data).get(),sizeof(DATATYPE)*im._n_max*im._n_max);
			memcpy((void *)_index.get(),(void *)(im._index).get(),sizeof(DIMTYPE)*im._n_max); // selected features (subspace) - index into _data

			if(im._permut)       {if(_n_max!=im._n_max) _permut.reset(new DIMTYPE[im._n_max]); memcpy((void *)_permut.get(),(void *)(im._permut).get(),sizeof(DIMTYPE)*im._n_max);} else _permut.reset();
			if(im._tmp_whereis)  {if(_n_max!=im._n_max) _tmp_whereis.reset(new DIMTYPE[im._n_max]); memcpy((void *)_tmp_whereis.get(),(void *)(im._tmp_whereis).get(),sizeof(DIMTYPE)*im._n_max);} else _tmp_whereis.reset();
			if(im._tmp_rowscale)	{if(_n_max!=im._n_max) _tmp_rowscale.reset(new DATATYPE[im._n_max]); memcpy((void *)_tmp_rowscale.get(),(void *)(im._tmp_rowscale).get(),sizeof(DATATYPE)*im._n_max);} else _tmp_rowscale.reset();

			_n_max=im._n_max;
		} else {
			_n_max=0;
			_data.reset();
			_index.reset();
			_permut.reset();
			_tmp_whereis.reset();
			_tmp_rowscale.reset();
		}
		// use assignment operator to accelerate...
		if(im._tmp_eq_roots) _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_roots))); // solve_equations result
			else _tmp_eq_roots.reset();
		if(im._tmp_eq_coefs)	_tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*(im._tmp_eq_coefs))); // equations' right hand side coefficients
			else _tmp_eq_coefs.reset();
	}
	return *this;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n_max, bool LUstorage)
{
	notify("Indexed_Matrix reset.");
	assert(n_max>0);
	if(n_max!=_n_max) {
		_n=_d=_n_max=n_max;
		_data.reset(new DATATYPE[_n_max*_n_max]);
		_index.reset(new DIMTYPE[_n_max]); // selected features (subspace) - index into _data
		if(LUstorage) {
			_permut.reset(new DIMTYPE[_n_max]); 
			_tmp_whereis.reset(new DIMTYPE[_n_max]); 
		} else {
			_permut.reset();
			_tmp_whereis.reset();
		}
		_tmp_rowscale.reset();
		_tmp_eq_coefs.reset();
		_tmp_eq_roots.reset();
	}
	_d=_n=_n_max;
	for(DIMTYPE i=0;i<_n_max;i++) _index[i]=i;
	if(LUstorage) {
		for(DIMTYPE i=0;i<_n_max;i++) _permut[i]=i;
		for(DIMTYPE i=0;i<_n_max;i++) _tmp_whereis[i]=i;
	}
	_sign=0;	
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::redim(const DIMTYPE n)
{
	notify("Indexed_Matrix redim.");
	assert(n>=0 && n<=_n_max);
	assert(_data);
	assert(_index);
	if(n==0) _d=_n=_n_max; else _d=_n=n;
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	if(_permut) for(DIMTYPE i=0;i<_n;i++) _permut[i]=i;
	if(_tmp_whereis) for(DIMTYPE i=0;i<_n;i++) _tmp_whereis[i]=i;
	_sign=0;
}

template<typename DATATYPE, typename DIMTYPE, typename SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub) // fill _index[] -> access to data is then mapped to sub-matrix
{
	assert(_n>=sub->get_d_raw()); //enough space to store subset ?
	assert(_index);
	DIMTYPE f;
	_d=0;
	for(bool b=sub->getFirstFeature(f);b!=false;b=sub->getNextFeature(f)) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::denarrow() // reset _index[] -> access is reset to the full matrix
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::set_all_raw_to(const DATATYPE val)
{
	assert(_data);
	for(DIMTYPE i=0;i<_n*_n;i++) _data[i]=val;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::add_raw(const Indexed_Matrix& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n*_n;i++) _data[i]+=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::add_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n*_n;i++) _data[i]+=mul * mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::copy_constmul_raw(const Indexed_Matrix& mr, const DATATYPE mul)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n*_n;i++) _data[i]=mul * mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::copy_raw(const Indexed_Matrix& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n*_n;i++) _data[i]=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::LUdecompose(Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &result)
{
	// NOTE: 'result' raw data access assumed (result._index ignored)
	//       while this->_index is used when accessing this->_data
	notify("Indexed_Matrix::LUdecompose");
	assert(result._n==_d); // enough pre-allocated space required
	assert(result._data);
	assert(result._permut);
	assert(result._tmp_whereis);
	assert(_index);
	result.denarrow(); // no indexes needed in result, will be written in raw ordering

	if(!_tmp_rowscale) _tmp_rowscale.reset(new DATATYPE[_n_max]); // create 'implicit scaling of rows' buffer once for the rest of Indexed_Matrix life

	DATATYPE max;
	DIMTYPE idxmax=-1;
	DATATYPE pom,sum;
	DIMTYPE j,i1,i2,tmpi;
	DIMTYPE pROW,pROWK,pCOL;

	result._d=_d;
	pROW=0;
	for(i1=0;i1<_d;i1++) { // copy source sub-matrix to result buffer for further manipulations
		for(i2=0;i2<_d;i2++) {
			assert(pROW+i2<_d*_d);
			assert(_index[i1]*_n+_index[i2]<_n*_n);
			result._data[pROW+i2]=_data[_index[i1]*_n+_index[i2]];
		}
		pROW+=_d;
	}

	result._sign=1;
	/* initialize permutation vector */
	for(j=0;j<_d;j++) {result._permut[j]=j; result._tmp_whereis[j]=j;}

	/* find scaling of rows */
	pROW=0;
	for(i1=0;i1<_d;i1++)
	{
		max=0.0;
		for(j=0;j<_d;j++) {assert(pROW+j<_d*_d); if((pom=fabs(result._data[pROW+j]))>max) max=pom;}

		if(max<std::numeric_limits<DATATYPE>::min()) throw fst_error("Indexed_Matrix::LUdecompose() value too small/matrix singular or near-singular",23);

		_tmp_rowscale[i1]=1.0/max;
		pROW+=_d;
	}

	/* Crout's method */
	//pCOL=0;
	for(j=0;j<_d;j++)
	{
		//pROW=0;
		for(i1=0;i1<j;i1++)
		{
			pROW=result._tmp_whereis[i1]*_d;
			assert(pROW+j<_d*_d);
			sum=result._data[pROW+j];

			//pROWK=0;
			for(i2=0;i2<i1;i2++)
			{
				pROWK=result._tmp_whereis[i2]*_d;
				assert(pROW+i2<_d*_d);
				assert(pROWK+j<_d*_d);
				sum-=result._data[pROW+i2]*result._data[pROWK+j];
				//pROWK+=_d;
			}
			result._data[pROW+j]=sum;
			//pROW+=_d;
		}
		/* search for largest pivot */
		max=0.0;
		//pROW=j*_d;
		for(i1=j;i1<_d;i1++)
		{
			pROW=result._tmp_whereis[i1]*_d;
			assert(pROW+j<_d*_d);
			sum=result._data[pROW+j];

			//pROWK=0;
			for(i2=0;i2<j;i2++)
			{
				pROWK=result._tmp_whereis[i2]*_d;
				assert(pROW+i2<_d*_d);
				assert(pROWK+j<_d*_d);
				sum-=result._data[pROW+i2]*result._data[pROWK+j];
				//pROWK+=_d;
			}
			result._data[pROW+j]=sum;
			pom=_tmp_rowscale[i1]*fabs(sum);
			if(pom>max) {max=pom; idxmax=i1;}
			//pROW+=_d;
		}

		/* interchange rows ? */
		if(j!=idxmax)
		{
			// NOTE: row swaps just marked in _tmp_whereis
			
			assert(idxmax>=0 && idxmax<_d);
			assert(j>=0 && j<_d);
			
			pom=_tmp_rowscale[idxmax];
			_tmp_rowscale[idxmax]=_tmp_rowscale[j];
			_tmp_rowscale[j]=pom;

			i1=result._tmp_whereis[j];
			i2=result._tmp_whereis[idxmax];

			assert(i1>=0 && i1<_d);
			assert(i2>=0 && i2<_d);

			tmpi=result._permut[i1];
			result._permut[i1]=result._permut[i2];
			result._permut[i2]=tmpi;

			tmpi=result._tmp_whereis[result._permut[i1]];
			result._tmp_whereis[result._permut[i1]]=result._tmp_whereis[result._permut[i2]];
			result._tmp_whereis[result._permut[i2]]=tmpi;

			result._sign=-result._sign;
		}

		pCOL=result._tmp_whereis[j]*_d;
		
		//if(fabs(result._data[pCOL+j])<std::numeric_limits<DATATYPE>::min()) {result._data[pCOL+j]=std::numeric_limits<DATATYPE>::min(); notify("NOTE: too small value set to minimum representable number.");}
		//if(fabs(result._data[pCOL+j])<_safedown) throw fst_error("result._data[] value too small."); //result._data[pCOL+j]=_safedown;

		if(j<(_d-1))
		{
			assert(pCOL+j<_d*_d);
			if(fabs(result._data[pCOL+j])<=std::numeric_limits<DATATYPE>::min()) {
				if(result._data[pCOL+j]<0.0) pom=-std::numeric_limits<DATATYPE>::max();
				else pom=std::numeric_limits<DATATYPE>::max(); 
				notify("NOTE: division by too small a number - result set to maximum representable number.");
			} else
			pom=1.0/result._data[pCOL+j];

			//pROW=(j+1)*_d;
			for(i1=j+1;i1<_d;i1++)
			{
				pROW=result._tmp_whereis[i1]*_d;
				assert(pROW+j<_d*_d);
				result._data[pROW+j] *= pom;
				//pROW+=_d;
			}
		}

		//pCOL+=_d;
	}	
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::solve_equations(Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &result_vec, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_matrix, Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> &coef_vec)
{
	// NOTE: result_vec, coef_vec, and LU_matrix raw data access assumed (*->_index ignored)
	//       while this->_index is used when accessing this->_data
	notify("Indexed_Matrix::solve_equations");
	assert(result_vec._data);
	assert(result_vec._n>=_d); // enough pre-allocated space required
	assert(coef_vec._data);
	assert(coef_vec._n>=_d); // enough pre-allocated space required
	assert(LU_matrix._n==_d);
	assert(LU_matrix._permut);
	assert(LU_matrix._tmp_whereis);

	DIMTYPE i,j;
	DATATYPE sum;
	DIMTYPE pROW;

	for(i=0;i<_d;i++) {assert(LU_matrix._tmp_whereis[i]<_d); result_vec[i]=coef_vec[LU_matrix._tmp_whereis[i]];} /* spravne serazeni b */

	//pROW=n;
	for(i=1;i<_d;i++)  /* result_vec[0] se nemeni */
	{
		pROW=LU_matrix._tmp_whereis[i]*_d;
		sum=0.0;
		for(j=0;j<i;j++) {
			assert(pROW+j<_d*_d);
			sum+=LU_matrix._data[pROW+j]*result_vec[j];
		}
		result_vec[i]-=sum;
		//pROW+=n;
	}

	//pROW=n*(n-1);
	for(i=_d;i>0;i--)
	{
		pROW=LU_matrix._tmp_whereis[i-1]*_d;
		assert(pROW+i-1<_d*_d);
		sum=0.0;
		for(j=i;j<_d;j++) sum+=LU_matrix._data[pROW+j]*result_vec[j];
		if(fabs(LU_matrix._data[pROW+i-1])<=std::numeric_limits<DATATYPE>::min()) {
			if(LU_matrix._data[pROW+i-1]*(result_vec[i-1]-sum)<0.0) result_vec[i-1]=-std::numeric_limits<DATATYPE>::max();
			else result_vec[i-1]=std::numeric_limits<DATATYPE>::max(); 
			notify("NOTE: division by too small number - result set to maximum representable number.");
		} else {
			result_vec[i-1]=(result_vec[i-1]-sum)/LU_matrix._data[pROW+i-1];
		}
		//pROW-=n;
	}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::invert(Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &result, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_Matrix)
{
	// NOTE: 'result' and 'LU_matrix' raw data access assumed (*->_index ignored)
	//       while this->_index is used when accessing this->_data
	notify("Indexed_Matrix::invert");
	assert(result._data);
	assert(result._n==_d); // enough pre-allocated space required
	assert(LU_Matrix._data);
	assert(LU_Matrix._permut);
	assert(LU_Matrix._index);
	assert(LU_Matrix._d==_d);
	
	if(_tmp_eq_roots) assert(_tmp_eq_roots->_n>=_d);
	else _tmp_eq_roots.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(_n_max)); // solve_equations result
	if(_tmp_eq_coefs) assert(_tmp_eq_coefs->_n>=_d);
	else _tmp_eq_coefs.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(_n_max)); // equations' right hand side coefficients

	DIMTYPE pROW;
	DIMTYPE i,j;

	for(j=0;j<_d;j++)
	{
		for(i=0;i<_d;i++) (*_tmp_eq_coefs)[i]=0.0;
		(*_tmp_eq_coefs)[j]=1.0;

		solve_equations(*_tmp_eq_roots,LU_Matrix,*_tmp_eq_coefs);

		pROW=0;
		for(i=0;i<_d;i++)
		{
			result._data[pROW+j]=(*_tmp_eq_roots)[i];
			pROW+=_d;
		}
	}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
DATATYPE Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>::determinant(const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET> &LU_matrix)
{
	// NOTE: 'LU_matrix' raw data access assumed (*->_index ignored)
	assert(LU_matrix._data);
	assert(LU_matrix._tmp_whereis);	
	assert(LU_matrix._d==_d);

	notify("Indexed_Matrix::determinant");
	DATATYPE det;
	DIMTYPE pROW; //,step;
	DIMTYPE j;
	//LUdecompose(*_tmp_lu_matrix);

	det=LU_matrix._sign;
	//pROW=0; step=_d+1;
	for(j=0;j<_d;j++)
	{
		pROW=LU_matrix._tmp_whereis[j]*_d+j;
		assert(pROW<_d*_d);
		det*=LU_matrix._data[pROW];
		//pROW+=step;
	}
	return det;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>& mr)
{
	assert(mr._n>0);
	assert(mr._d>0);
	assert(mr._data);
	assert(mr._index);
	os << "----------------"<<endl;
	os << "Indexed_Matrix::";
	os << "_index: "; for(DIMTYPE j=0;j<mr._d;j++) {os << mr._index[j] << " ";} os<<endl;
	os << "_data (indexed):"<<endl;
	for(DIMTYPE i=0;i<mr._d;i++) {for(DIMTYPE j=0;j<mr._d;j++) {os << mr._data[mr._index[i]*mr._n+mr._index[j]] << " ";} os << endl;}
	os << "----------------"<<endl;
	return os;
}

} // namespace
#endif // FSTINDEXEDMATRIX_H ///:~
