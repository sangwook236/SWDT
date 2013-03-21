#ifndef FSTDATAFILETRN_H
#define FSTDATAFILETRN_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_file_TRN.hpp
   \brief   Enables data accessor objects to read TRN type of data files
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    October 2010
   \version 3.0.1.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44),
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/, 
                tested with version 3.00)
   \note    Note that LibSVM is required for SVM related tools only,
            as demonstrated in demo12t.cpp, demo23.cpp, demo25t.cpp, demo32t.cpp, etc.

   \warning Please note that this code plays only supportive role
            and does not represent coding standards followed in core
            FST3 codes.
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
#include <vector>
#include <cstdio>
#include <cstring>
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

/*! \brief TRN data format filter
    \note USES OBSOLETE CODE TO ACCESS *.TRN FILES */
template<typename DATATYPE, typename IDXTYPE>
class Data_File_TRN {
public:
	Data_File_TRN() {_filename=""; _title=""; _pocpriz=0; _poctrid=0; _valid=false; dov=NULL;}
	~Data_File_TRN() {close();}
	void open(const string& filename); //!< opens TRN file and reads header
	void close() {if(dov!=NULL) {fclose(dov); dov=NULL;}}; //!< closes TRN file if it is open
	const unsigned int getNoOfFeatures() const {return _pocpriz;}
	const unsigned int getNoOfClasses() const {return _poctrid;}
	const IDXTYPE getClassSize(const unsigned int cidx) const {if(cidx<_poctrid) return _pocprv[cidx]; else return 0;}
	bool read_val(DATATYPE& target); // does not throw
	DATATYPE read_val();// throws
private:
	Data_File_TRN(const Data_File_TRN&); // no definition -> prevent passing by value
protected:
	void GetFileParams();
	string _filename;	/* jmeno vstupniho datoveho souboru */
	string _title;	/* nepovinny nazev dat (soucast dat.souboru) */
	unsigned int _pocpriz;	/* viz nize */
	unsigned int _poctrid;	/* viz nize */
	std::vector<IDXTYPE> _pocprv;	/* pozor na dvoji ukazatel na tataz data */
	IDXTYPE _datastart;	/* pocatek dat za hlavickou */
	int _valid;	//!< indicates the state, when header is successfully read and file open and ready to be read for actual data.
	FILE *dov;
};

template<typename DATATYPE, typename IDXTYPE>
void Data_File_TRN<DATATYPE,IDXTYPE>::open(const string& filename)
{
	_filename=filename;
	_pocprv.clear();
	_pocpriz=0; _poctrid=0;
	GetFileParams();
}

template<typename DATATYPE, typename IDXTYPE>
void Data_File_TRN<DATATYPE,IDXTYPE>::GetFileParams()
{
  const unsigned int ONUMCOM=5;        // max no. of header keywords
  const unsigned int OHEADSTRLEN=1000; // max header line size
  const unsigned int OFILENAMELENGTH=500;
  const string ONOTITLE="no title";
  //FILE *dov;
  const char *command[ONUMCOM]={"#datafile",
			"#title",
			"#features",
			"#classes",
			"#data"};
  char s[OHEADSTRLEN];   /* sem vzdy nacte celou radku */
  char poms[OHEADSTRLEN];/* a sem si prekopiruje prvni slovo te radky */
  char *s2;             /* tim si ukaze na zacatek prikazu, je-li tam */
  char *s3;             /* tim si ukazuje na data za prikazem */
  int error=0;
  int empty=0;          /* prazdny soubor ? z Optimy  - FALSE */
  int hotov=0;          /* indikator precteni cele hlavicky */
  unsigned int i;
  int pom;
  int stav=0;		/* indikator vyskytu jednotlivych prikazu v hlav. */
  int prikaz=0;         /* cislo rozpoznaneho prikazu */
  int citac;		/* bitova maska pro zapis a cteni bitu z stav */

  _valid = false;
  //info->classifyvalid = false;
  // test chyby
  if((dov=fopen(_filename.c_str(),"r"))==NULL) {throw fst_error("Data_File_TRN file read problem.",1);}
  empty=1;  /* TRUE - prazdny soubor */
  _title=ONOTITLE;

  while((!error)&&(!feof(dov))&&(!hotov))
  {
    _datastart=ftell(dov);
    fgets(s,OHEADSTRLEN,dov);

    pom=sscanf(s,"%s",poms); 			/* ? precet vubec neco ? */
    if((pom)&&(pom!=EOF)&&(poms[0]!=';')&&(poms[0]!=0))
    {
      empty=0; /* FALSE */
      citac=1;
      prikaz=0;
      s3=NULL;
      for(i=0;(i<ONUMCOM)&&(!prikaz);i++)         /* ? je to v *command[] ? */
      {
        s2=strstr(poms,command[i]);
        if(s2!=NULL)
        {
          prikaz=citac;
          s3=strstr(s,poms)+strlen(command[i]); /* s3 ukazuj do s za slovo */
        }
        citac<<=1;
      }
      if(s3==NULL)
      {
        // test chyby
        if(!((stav&1)&&(stav%4))) // puvodne i stav&8
           {throw fst_error("Data_File_TRN file read problem.",10);}
      }
      else for(i=0;s3[i]!=0;i++)if(s3[i]==',')s3[i]=' ';

      // test chyby "tohle uz tu bylo"
      if((prikaz)&&(stav&prikaz)) {throw fst_error("Data_File_TRN file read problem.",5);}
      else
      if(!error)
      {
        switch(prikaz)
        {              // test chyby
          case 0:  if((stav&1)&&(stav&4)) /*puvodne i stav&8=pocet trid, kvuli klasifikaci povoleno bez poctu trid*/	/*data (bez prikazu)*/
               {
                 hotov=1;
                     /*  fseek(dov,info->datastart,SEEK_SET);  */ 
               }
               else {throw fst_error("Data_File_TRN file read problem.",6);}
               break;
                       // test chyby
          case 2:  if(!(stav&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                       /* title */
                 s2=strstr(s3,"\n"); if(s2!=NULL) s2[0]=0; // vymaz newline
                 while((s3[0]!=0)&&((s3[0]==' ')||(s3[0]=='\t'))) s3++;
                 // test chyby
                 if(strlen(s3)>=OFILENAMELENGTH) // osekni s3
                     s3[OFILENAMELENGTH-1]=0;
                 //strcpy(_title,s3);
                 _title=s3;
               }
               break;
                       // test chyby
          case 4:  if(!(stav&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                       /* features */
                 pom=sscanf(s3,"%d",&(_pocpriz));
                 // test chyby
                 if((!pom)||(pom==EOF))
                 {
                   _pocpriz=-1;
                   throw fst_error("Data_File_TRN file read problem.",7);
                 }
               }
               break;
                       // test chyby
          case 8:  if(!(stav&1)) {throw fst_error("Data_File_TRN file read problem.",6);} 
               else {                                                        /*classes */
                 pom=sscanf(s3,"%d",&(_poctrid));
                 // test chyby
                 if((!pom)||(pom==EOF))
                 {
                   _poctrid=-1;
                   throw fst_error("Data_File_TRN file read problem.",8);
                 }
                 else
                 {
                   if(_poctrid>0)
                   {
                   	_pocprv.clear();
                   }
                   sscanf(s3,"%s",poms);
                   s3=strstr(s3,poms)+strlen(poms);
                   for(i=0;(i<_poctrid)&&(!error);i++)
                   {
                   	IDXTYPE temp;
                     pom=sscanf(s3,"%d",&(temp));
                     // test chyby
                     if((!pom)||(pom==EOF))
                     {
                       throw fst_error("Data_File_TRN file read problem.",9);
                     }
                     else
                     {
                       _pocprv.push_back(temp);
                       sscanf(s3,"%s",poms);
                       s3=strstr(s3,poms)+strlen(poms);
                     }
                   }
                 }
               }
               break;
                       // test chyby    
          case 16: if((stav&1)&&(stav&4))	/*data*/
               {
                 hotov=1;
                 _datastart=ftell(dov);
               }
               else  {throw fst_error("Data_File_TRN file read problem.",6);} 
               break;
          default: ;
        }
        stav|=prikaz;
      }
    }
  }
  // test chyby   "prazdny soubor"
  if(empty) {throw fst_error("Data_File_TRN file read problem.",11);}

  if(!(stav&8)) // nedef. pocet trid
  {
      _poctrid = 0; // nedefinovano
      throw fst_error("Data_File_TRN file read problem.",8); // vrat "chybovou" hlaskou, ze to neni obvykly soubor
  }
  if(_poctrid!=_pocprv.size()) {throw fst_error("Data_File_TRN file read problem.",88);}
  if(_poctrid<1) {throw fst_error("Data_File_TRN file read problem.",100);}
  if(_pocpriz<1) {throw fst_error("Data_File_TRN file read problem.",101);}
  for(i=0;i<_poctrid;i++) if(_pocprv[i]<1) {throw fst_error("Data_File_TRN file read problem.",102);}
 
  _valid = true;
}

template<typename DATATYPE, typename IDXTYPE>
bool Data_File_TRN<DATATYPE,IDXTYPE>::read_val(DATATYPE& target)
{
	if(_valid) {
		double dbltmp; int err;
		err=fscanf(dov,"%lf",&dbltmp); 
		if(err>0) {
			target=dbltmp;
			return true;
		}
	}
	return false;
}

template<typename DATATYPE, typename IDXTYPE>
DATATYPE Data_File_TRN<DATATYPE,IDXTYPE>::read_val()
{
	if(_valid) {
		double dbltmp; int err;
		err=fscanf(dov,"%lf",&dbltmp); 
		if(err>0) return DATATYPE(dbltmp); else throw fst_error("Data_File_TRN::read_val() problem.",200);
	}
	throw fst_error("Data_File_TRN::read_val() problem.",201);
}

} // namespace
#endif // FSTDATAFILETRN_H ///:~
