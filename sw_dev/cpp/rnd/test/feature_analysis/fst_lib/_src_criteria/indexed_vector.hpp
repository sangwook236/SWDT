#ifndef FSTINDEXEDVECTOR_H
#define FSTINDEXEDVECTOR_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    indexed_vector.hpp
   \brief   Vector representation and operations, allows operation in selected subspace
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
#include "indexed_matrix.hpp"

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
class Indexed_Matrix;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Vector;

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& tdb);

//! Vector representation and operations, allows operation in selected subspace
template<typename DATATYPE, typename DIMTYPE, class SUBSET>
class Indexed_Vector { //!< helper structure for passing matrix op intermediate data between higher-level matrix objects and functions
public:
	typedef boost::shared_ptr<SUBSET> PSubset;
	Indexed_Vector() : _n_max(0), _n(0), _d(0) {notify("Indexed_Vector (empty) constructor");}
	Indexed_Vector(const DIMTYPE n)  : _n_max(0), _n(0), _d(0) {notify("Indexed_Vector constructor"); reset(n);}
	Indexed_Vector(const Indexed_Vector& iv); // copy-constructor
	Indexed_Vector& operator=(Indexed_Vector const &iv);
	~Indexed_Vector() {notify("Indexed_Vector destructor");}
	void reset(const DIMTYPE n_max); // reallocates
	void redim(const DIMTYPE n=0); // keeps main allocations, just resets index arrays and adjusts _n; n==0 here equals _n_max
	
	DATATYPE& operator[](const DIMTYPE i) {assert(i>=0 && i<_n); assert(_data); return _data[i];} // raw access
	DATATYPE& at_raw(const DIMTYPE i) {assert(i>=0 && i<_n); assert(_data); return _data[i];} // raw access
	DATATYPE& at(const DIMTYPE i) {assert(i>=0 && i<_d); assert(_index); return _data[_index[i]];} // soft (indexed) access
	DATATYPE* get_raw() {return _data.get();} //&(_data[0])

	void narrow_to(const PSubset sub); // fill _index[] -> access to data is then mapped to sub-vector
	void denarrow(); // reset _index[] -> access is reset to the full vector
	DIMTYPE get_n() const {return _n;}
	DIMTYPE get_d() const {return _d;}

	// "raw" operations, ignoring index[]
	void set_all_raw_to(const DATATYPE val);
	void subtract_raw(const Indexed_Vector& vr);
	void copy_raw(const Indexed_Vector& vr);

	friend class Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>;
	friend std::ostream& operator<< <>(std::ostream& os, const Indexed_Vector& tdb);
	
protected:
	// NOTE: vector size is defined by:
	//       _n_max - hard constant representing maximum allocated space of size _n_max
	//       _n     - "full feature set" dimensionality, _n<=_n_max
	//       _d     - "virtual subset" dimensionality, _d<=_n
	//       (this logic replicates that in Indexed_Matrix, where it is more important)
	DIMTYPE _n_max; // allocated space can hold _n_max*_n_max matrix
	DIMTYPE _n; // full feature set dimensionality
	boost::scoped_array<DATATYPE> _data; // raw vector data
	// sub-vector access tools
	DIMTYPE _d; // virtual sub-vector dimensionality
	boost::scoped_array<DIMTYPE> _index; // maps virtual sub-vector _d dimensions to original _n dimensions
};

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::Indexed_Vector(const Indexed_Vector& iv) : // copy-constructor
	_n_max(iv._n_max),
	_n(iv._n),
	_d(iv._d)
{
	notify("Indexed_Vector copy-constructor.");
	if(_n_max>0)
	{
		_data.reset(new DATATYPE[_n_max]);	memcpy((void *)_data.get(),(void *)(iv._data).get(),sizeof(DATATYPE)*_n_max);
		_index.reset(new DIMTYPE[_n_max]); memcpy((void *)_index.get(),(void *)(iv._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
	}
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::operator=(Indexed_Vector const &iv)
{
	notify("Indexed_Vector operator=.");
	if (this != &iv)
	{
		_n=iv._n;
		_d=iv._d;
		if(_n_max!=iv._n_max) 
		{
			_n_max=iv._n_max;
			if(_n_max>0)
			{
				_data.reset(new DATATYPE[_n_max]);	
				_index.reset(new DIMTYPE[_n_max]); 
			} else {
				assert(!iv._data);
				assert(!iv._index);
				_data.reset();
				_index.reset();
			}
		}
		if(_n_max>0)
		{
			memcpy((void *)_data.get(),(void *)(iv._data).get(),sizeof(DATATYPE)*_n_max);
			memcpy((void *)_index.get(),(void *)(iv._index).get(),sizeof(DIMTYPE)*_n_max); // selected features (subspace) - index into _data
		}
	}
	return *this;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::reset(const DIMTYPE n_max)
{
	notify("Indexed_Vector reset.");
	assert(n_max>0);
	if(n_max!=_n_max) {
		_n=_d=_n_max=n_max;
		_data.reset(new DATATYPE[_n_max]);
		_index.reset(new DIMTYPE[n_max]); // selected features (subspace) - index into _data
	}
	for(DIMTYPE i=0;i<_n_max;i++) _index[i]=i;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::redim(const DIMTYPE n)
{
	notify("Indexed_Vector redim.");
	assert(n>=0 && n<=_n_max);
	assert(_data);
	assert(_index);
	if(n==0) _d=_n=_n_max; else _d=_n=n;
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::narrow_to(const PSubset sub) // fill _index[] -> access to data is then mapped to sub-matrix
{
	assert(_n>=sub->get_d_raw()); //enough space to store subset ?
	assert(_index);
	DIMTYPE f;
	_d=0;
	for(bool b=sub->getFirstFeature(f);b!=false;b=sub->getNextFeature(f)) _index[_d++]=f;
	assert(_d>0);
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::denarrow() // reset _index[] -> access is reset to the full matrix
{
	assert(_index);
	for(DIMTYPE i=0;i<_n;i++) _index[i]=i;
	_d=_n;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::set_all_raw_to(const DATATYPE val)
{
	assert(_data);
	for(DIMTYPE i=0;i<_n;i++) _data[i]=val;
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::subtract_raw(const Indexed_Vector& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n;i++) _data[i]-=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
void Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>::copy_raw(const Indexed_Vector& mr)
{
	assert(_data);
	assert(mr._data);
	assert(mr.get_n()==_n);
	for(DIMTYPE i=0;i<_n;i++) _data[i]=mr._data[i];
}

template<typename DATATYPE, typename DIMTYPE, class SUBSET>
std::ostream& operator<<(std::ostream& os, const Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>& tdb)
{
	assert(tdb._n>0);
	assert(tdb._d>0);
	assert(tdb._data);
	assert(tdb._index);
	os << "----------------"<<endl;
	os << "Indexed_Vector::" << endl;
	os << "_index: "; for(DIMTYPE j=0;j<tdb._d;j++) {os << tdb._index[j] << " ";} os<<endl;
	os << "_data:  "; for(DIMTYPE j=0;j<tdb._d;j++) {os << tdb._data[j] << " ";} os << endl;
	os << "----------------"<<endl;
	return os;
}


} // namespace
#endif // FSTINDEXEDVECTOR_H ///:~
