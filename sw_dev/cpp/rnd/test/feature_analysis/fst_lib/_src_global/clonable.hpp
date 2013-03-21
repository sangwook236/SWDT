#ifndef FSTCLONABLE_H
#define FSTCLONABLE_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    clonable.hpp
   \brief   Defines interface for classes that may need to get cloned
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

#include <iostream>
#include "error.hpp"
#include "global.hpp"

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

class Clonable;

std::ostream& operator<<(std::ostream& os, const Clonable& cl);

/*! \brief Abstract class, defines interface for classes that may need to get cloned
    
    (defined especially to support threading in Sequential_Step specializations)
    \note Note that clonable objects may appear in threaded code - their output
    thus should be guarded using syncout class (see global.*pp)
*/  
class Clonable { // abstract class
	friend std::ostream& operator<<(std::ostream& os, const Clonable& cl);
public:
	Clonable(): _clone(false), _sharing(false) {}
	virtual ~Clonable() {}

	/*! create 1:1 independent clone of the current object */
	virtual Clonable* clone() const =0; //{throw fst_error("Clonable::clone() not implemented.");}
	
	/*! create equivalent clone of the current object,
	    parmitting read-only access to structures in the source
	    object (allows referencing instead of copying of large memory structures).
	    may be faster and save space but requires more caution with respect to concurrency 
	    Use example: Data_Accessor memory data representation cloning */
	virtual Clonable* sharing_clone() const =0; //{throw fst_error("Clonable::sharing_clone() not implemented.");}

	/*! create clone of the current object, ignoring
	    internal temporary structures to save speed.
	    Does not replicate exact object state. The clone
	    must be used carefully in a way that ensures
	    internal structure re-initialization
	    Use example: Data_Splitter cloning or
	    Classifier_SVM cloning due to inability to clone
	    external structures defined in LibSVM */
	virtual Clonable* stateless_clone() const =0; //{throw fst_error("Clonable::stateless_clone() not implemented.");}

	/*! check whether this instance is a clone */
	bool is_clone() const {return _clone;}
	bool is_sharing() const {return _sharing;}
	bool is_sharing_clone() const {return _sharing && _clone;}
	
	virtual std::ostream& print(std::ostream& os) const {return os;};
protected:	
	void set_cloned(const bool cloned=true) {_clone=cloned;}
	void set_sharing(const bool sharing=true) {_sharing=sharing;}
	void set_sharing_cloned(const bool cloned=true, const bool sharing=true) {_clone=cloned; _sharing=sharing;}
private:
	/*! cloning indicator, should be set inside clone() and weak_clone() */
	bool _clone;
	bool _sharing;
};

std::ostream& operator<<(std::ostream& os, const Clonable& cl) {
	return cl.print(os);
}

} // namespace
#endif // FSTCLONABLE_H ///:~
