#ifndef FSTCRITERIONNORMALDIVERGENCE_H
#define FSTCRITERIONNORMALDIVERGENCE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    criterion_normal_divergence.hpp
   \brief   Implements Divergence (distance) based on normal (gaussian) model to serve as feature selection criterion
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
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz)
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
#include <cmath>
#include "error.hpp"
#include "global.hpp"
#include "criterion_normal.hpp"
#include "indexed_vector.hpp"
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

/*! \brief Implements Divergence (distance) based on normal (gaussian) model to serve as feature selection criterion 
    \note Defined for two-class problems only */
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal_Divergence : public Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> {
public:
	typedef Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> parent;
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Criterion_Normal_Divergence() {notify("Criterion_Normal_Divergence constructor.");}
	virtual ~Criterion_Normal_Divergence() {notify("Criterion_Normal_Divergence destructor.");}

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub);
	virtual bool initialize(PDataAccessor da); 

	Criterion_Normal_Divergence* clone() const;
	Criterion_Normal_Divergence* sharing_clone() const {throw fst_error("Criterion_Normal_Divergence::sharing_clone() not supported, use Criterion_Normal_Divergence::clone() instead.");}
	Criterion_Normal_Divergence* stateless_clone() const {throw fst_error("Criterion_Normal_Divergence::stateless_clone() not supported, use Criterion_Normal_Divergence::clone() instead.");}
	
	virtual std::ostream& print(std::ostream& os) const {os << "Criterion_Normal_Divergence()"; return os;}
private:
	Criterion_Normal_Divergence(const Criterion_Normal_Divergence& cnd); // copy-constructor
private:
	boost::scoped_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _meandif; // to store mean[i]-mean[j]
	boost::scoped_ptr<Indexed_Vector<DATATYPE,DIMTYPE,SUBSET> > _diag; // 
	boost::scoped_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf1; // to store matrix inversions
	boost::scoped_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _invbuf2; // to store matrix inversions
	boost::scoped_ptr<Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> > _LUtemp; // to store matrix LU decomposition
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal_Divergence(const Criterion_Normal_Divergence& cnd) :
	Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(cnd)
{
	notify("Criterion_Normal_Divergence copy-constructor.");
	if(cnd._meandif) _meandif.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*cnd._meandif));
	if(cnd._diag) _diag.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>(*cnd._diag));
	if(cnd._invbuf1) _invbuf1.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnd._invbuf1));
	if(cnd._invbuf2) _invbuf2.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnd._invbuf2));
	if(cnd._LUtemp) _LUtemp.reset(new Indexed_Matrix<DATATYPE,DIMTYPE,SUBSET>(*cnd._LUtemp));
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>* Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::clone() const
{
	Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> *clone=new Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*this);
	clone->set_cloned();
	return clone;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::evaluate(RETURNTYPE &result, PSubset sub)
{
	notify("Criterion_Normal_Divergence::evaluate().");
	assert(parent::_model);
	assert(parent::get_n()>0);
	assert(_meandif);
	assert(_diag);
	assert(_invbuf1);
	assert(_invbuf2);
	assert(_LUtemp);
	assert(parent::get_n()==_meandif->get_n());
	assert(parent::get_n()==_diag->get_n());
	assert(parent::get_n()<=_invbuf1->get_n_max());
	assert(parent::get_n()<=_invbuf2->get_n_max());
	assert(parent::get_n()<=_LUtemp->get_n_max());
	assert(sub);
	
	if(sub->get_d_raw()==0) return false;

	parent::_model->narrow_to(sub);
	_meandif->narrow_to(sub);
	
	_invbuf1->redim(parent::get_d());
	_invbuf2->redim(parent::get_d());
	_LUtemp->redim(parent::get_d());

	Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> &_cov1=(parent::_model->get_cov()[0]);
	Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET> &_cov2=(parent::_model->get_cov()[1]);

	_cov1.LUdecompose(*_LUtemp);
	_cov1.invert(*_invbuf1,*_LUtemp);	
	_cov2.LUdecompose(*_LUtemp);
	_cov2.invert(*_invbuf2,*_LUtemp);

	RETURNTYPE left=0.0;
	RETURNTYPE pom;
	for(DIMTYPE j=0;j<parent::get_d();j++)
	{
		pom=0.0;
		for(DIMTYPE i=0;i<parent::get_d();i++)
		{
			pom+=_meandif->at(i)*(_invbuf1->at_raw(i,j)+_invbuf2->at_raw(i,j));
		}
		left+=pom*_meandif->at(j);
	}

	RETURNTYPE right=0.0;
	for(DIMTYPE i=0;i<parent::get_d();i++)
	{
		_diag->at_raw(i)=0.0;
		for(DIMTYPE k=0;k<parent::get_d();k++)
		{
			_diag->at_raw(i)+=_invbuf1->at_raw(i,k)*_cov2.at(k,i);
		}
	}
	for(DIMTYPE i=0;i<parent::get_d();i++)
	{
		for(DIMTYPE k=0;k<parent::get_d();k++)
		{
			_diag->at_raw(i)+=_invbuf2->at_raw(i,k)*_cov1.at(k,i);
		}
	}
	for(DIMTYPE i=0;i<parent::get_d();i++) right+=_diag->at_raw(i)-2.0;

	result=0.5*left+0.5*right;
	return true;
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal_Divergence<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	if(da->getNoOfClasses()!=2) throw fst_error("Criterion_Normal_Divergence() defined for two-class problems only.");
	notify("Criterion_Normal_Divergence::initialize().");
	parent::initialize(da);
		
	if(!_meandif) _meandif.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>);
	if(parent::get_n()>_meandif->get_n()) _meandif->reset(parent::get_n());
	if(!_diag) _diag.reset(new Indexed_Vector<DATATYPE,DIMTYPE,SUBSET>);
	if(parent::get_n()>_diag->get_n()) _diag->reset(parent::get_n());
	if(!_invbuf1) _invbuf1.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(parent::get_n()));
	if(parent::get_n()>_invbuf1->get_n()) _invbuf1->reset(parent::get_n());
	if(!_invbuf2) _invbuf2.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(parent::get_n()));
	if(parent::get_n()>_invbuf2->get_n()) _invbuf2->reset(parent::get_n());
	if(!_LUtemp) _LUtemp.reset(new Indexed_Matrix<REALTYPE,DIMTYPE,SUBSET>(parent::get_n(),true));
	if(parent::get_n()>_LUtemp->get_n()) _LUtemp->reset(parent::get_n(),true);

	_meandif->copy_raw(parent::_model->get_mean()[0]);
	_meandif->subtract_raw(parent::_model->get_mean()[1]);
		
	return true; }


} // namespace
#endif // FSTCRITERIONNORMALDIVERGENCE_H ///:~
