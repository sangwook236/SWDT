#ifndef FSTDATAFILEARFF_H
#define FSTDATAFILEARFF_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    data_file_ARFF.hpp
   \brief   Enables data accessor objects to read ARFF (Weka) type of data files
   \author  Jan Hora (hora@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    September 2012
   \version 3.1.1.beta
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

#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>
#include "error.hpp"
#include "global.hpp"

using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace FST {

	//////////////////////////////////////////////   Local utilities


	/*! \brief ARFF data format filter - representation of an attribute
	\note USES OBSOLETE CODE TO ACCESS *.ARFF FILES */
	class Data_File_ARFF_Feature {
	public:
		static const int TYPE_NUMERIC = 1;
		static const int TYPE_NOMINAL = 2;
		static const int TYPE_STRING = 3;

		string name;
		int type;
		int offset;
		int size;
		bool isClass;
		vector<string> values;			//  values for nominal type
	};

	/*! \brief ARFF data format filter - representation of a record
	\note USES OBSOLETE CODE TO ACCESS *.ARFF FILES */
	class Data_File_ARFF_Record {
		void *data;
		int classId;

	public:
		Data_File_ARFF_Record(int recordSize) {this->data = malloc(recordSize); classId = -1;}
		~Data_File_ARFF_Record();

		void *getData() { return this->data;};
		int getClassId() { return this->classId;};
		void setClassId(int pId) {this->classId = pId;};
	};

	Data_File_ARFF_Record::~Data_File_ARFF_Record() {
		if (this->data != NULL) free(this->data);
		this->data = NULL;
	}



	/////////////////////////////////////////////////    Main data_file_arff class
	/*! \brief ARFF data format filter
	\note USES OBSOLETE CODE TO ACCESS *.ARFF FILES */
	class Data_File_ARFF {
	private:
		string _fileName;
		vector<Data_File_ARFF_Record*> data;
		vector<Data_File_ARFF_Feature*> attributes;
		vector<Data_File_ARFF_Feature*> features;
		int *classSize;

		string relationName;
		int attributesCount;
		int classAttribute;
		int classCount;

		int recordSize;
		bool loaded;
	public:
		Data_File_ARFF(string fileName);
		~Data_File_ARFF();

		bool prepareFile();				// prepares the file contents

		void sortRecordsByClass();

		unsigned int getNoOfClasses();			// resturn number of classes
		unsigned int getNoOfFeatures() {return features.size();};			// returns number of features. 
		unsigned int getNoOfAttributes() {return attributes.size();};			// returns number of features. 
		unsigned int getTotalSize();				// returns the total number of records
		unsigned int getClassSize(int clsId);	// returns number of records for given class
		string getClassName(int clsId); // returns the nam of given class

		Data_File_ARFF_Record* getRecordAt(long i) {if (i<0 || i>= (long)data.size()) return NULL; else return data[i];};
		//  returns the value of given feature of given record
		float getFeatureValue(long record, int feature) {return *(float *)(((char *)(data[record]->getData()))+features[feature]->offset);};

		void printInfo();
	private:
		Data_File_ARFF_Feature *createAttribute(char *info);
		bool readNextRow(char *row);
		void prepareAttributes();
		Data_File_ARFF_Record *readNormalRow(char *row);
		Data_File_ARFF_Record *readSparseRow(char *row);
		bool readAttrValue(Data_File_ARFF_Feature *feature, Data_File_ARFF_Record *rec, char *value);

		///  trims a string (removes spaces, tabs, and cr / lf
		char *trim( char *szSource ) {
			char *pszEOS = szSource + strlen( szSource ) - 1;
			while( pszEOS >= szSource && (*pszEOS == ' ' || *pszEOS == 9 || *pszEOS == 13 || *pszEOS == 10)) *pszEOS-- = '\0';
			pszEOS = szSource;
			while( *pszEOS == ' ' || *pszEOS == 9  || *pszEOS == 13 || *pszEOS == 10) pszEOS++;
			return pszEOS;
		};

		///  replaces nonstandard strupr
		char *mystrupr( char *szSource ) {
			char *pszEOS = szSource;
			while( *pszEOS != 0 ) {*pszEOS=toupper(*pszEOS); pszEOS++;}
			return szSource;
		};
	};


	Data_File_ARFF::Data_File_ARFF(string fileName) {
		this->_fileName = fileName;
		this->classSize = NULL;
		attributesCount = 0;
		classAttribute = -1;
		classCount = -1;
		loaded = false;
	};

	Data_File_ARFF::~Data_File_ARFF() {
		for (unsigned i=0;i<attributes.size(); i++) delete attributes[i];
		this->attributes.clear();
		this->features.clear();
		for (unsigned i=0;i<data.size(); i++) delete data[i];
		this->data.clear();
		if (classSize != NULL) delete classSize;
		classSize = NULL;
	}

	bool Data_File_ARFF::prepareFile() {
		FILE *f =fopen(_fileName.c_str(),"rt");
		if(f==NULL) {throw FST::fst_error("Data_File_ARFF file read problem.",1);}//{perror(_fileName.c_str()); throw "Data_File_ARFF file read problem.";}
		char *buff = new char[8096];

		bool readingData = false;
		bool isError = false;
		int rowsReaded = 0;

		while (!feof(f) && !isError) {
			if (fgets(buff, 8096, f)==NULL) break;
			trim(buff);

			if (!buff[0]) continue;		//  empty row
			if (buff[0]=='%') continue;			// comment
			if (readingData) {
				if (readNextRow(buff)) rowsReaded++;
				else isError = true;
			} else if (buff[0]=='@') {
				//  get the second part of string
				char *val = buff+2; while(*val!=' ' && *val!=0 && *val!=9) val++; val[0] = 0; val++;	
				trim(mystrupr(buff));

				if (strcmp(buff, "@RELATION")==0) this->relationName = val;
				else if (strcmp(buff, "@DATA")==0) {
					readingData = true;
					prepareAttributes();
				} else if (strcmp(buff, "@ATTRIBUTE")==0) {
					Data_File_ARFF_Feature *feature = this->createAttribute(val);
					if (feature != NULL) {		//  feature created
						this->attributes.push_back(feature);
						this->attributesCount ++;
					} else {					//  something went wrong
						cout << "Attribute description not understood : " << buff << " " << val << endl;
						isError = true; break;
					}
				}
			} else {
				//  the row should be empty. Otherwise something is wrong..
				if (trim(buff)[0] != 0) {
					cout << "Error reading file - unexpected row : " << buff << endl;
					//isError = true; break;
				}
			}

		}

		loaded = !isError;
		cout <<  "Relation: " << this->relationName.c_str() << endl;
		cout <<  "Classes count: " << getNoOfClasses() << endl;
		cout <<  "Records count: " << rowsReaded << endl;

		delete buff;
		
		return !isError;
	}

	void Data_File_ARFF::printInfo() {
		if (!loaded) {cout << "File not loaded" << endl; return;};
		///////////  Debug info
		for (unsigned d=0; d < data.size(); d++) {
			char *dt = (char *)data[d]->getData();
			cout << "Record: cls=" << data[d]->getClassId() << " " ;
			for (unsigned i=0; i < attributes.size(); i++) {
				char *v = dt + attributes[i]->offset;
				if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NOMINAL) 
					cout << " " << *(short *)v;
				if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NUMERIC) 
					cout << " " << *(float*)v;
			}
			cout << endl;
		}

		cout <<  "Relation: " << this->relationName.c_str() << endl;
		cout <<  "Feature count: " << this->attributes.size() << endl;

		for (unsigned i=0; i < attributes.size(); i++) {
			cout << "  " << attributes[i]->name.c_str() << " : " << attributes[i]->type << endl;
			if (attributes[i]->type == Data_File_ARFF_Feature::TYPE_NOMINAL) {
				vector<string> vals = attributes[i]->values;
				for (unsigned j=0; j < vals.size(); j++) 
					cout << "    - " << vals[j].c_str() << endl;
			}
		}

		cout <<  "Classes count: " << getNoOfClasses() << " sizes: ";
		for (unsigned int i=0; i<getNoOfClasses() ; i++)
			cout << getClassName(i).c_str() << ": " << getClassSize(i) << "     ";
		cout << endl;


		cout <<  "Records count: " << data.size() << endl;

	}

	void Data_File_ARFF::prepareAttributes() {
		int size = 0;
		this->classAttribute = -1;

		for (unsigned fi = 0; fi < attributes.size(); fi ++ ) {
			Data_File_ARFF_Feature *feature = attributes[fi];
			feature->offset = size;
			size += feature->size;

//--S [] 2013/03/21: Sang-Wook Lee
#if defined(_MSC_VER)
			std::vector<char> tmp_vec(feature->name.length()+1);
			char *tmp = &tmp_vec[0];
#else
//--E [] 2013/03/21: Sang-Wook Lee
			char tmp[feature->name.length()+1];
//--S [] 2013/03/21: Sang-Wook Lee
#endif
//--E [] 2013/03/21: Sang-Wook Lee
			strcpy(tmp, feature->name.c_str());
			mystrupr(tmp);
			if ((strcmp(tmp, "CLASS")==0 || strcmp(tmp, "'CLASS'")==0) && feature->type==Data_File_ARFF_Feature::TYPE_NOMINAL) {
				this->classAttribute = fi;
				feature->isClass = true;
			} else {
				feature->isClass = false;
				features.push_back(feature);
			}
		}

		if (classAttribute>=0) {
			classCount = attributes[classAttribute]->values.size();
			classSize = new int[classCount];
			for (int i=0;i<classCount;i++) classSize[i] = 0;
			cout << "Classes count : " << classCount << endl;
		}
		this->recordSize = size;
	}

	Data_File_ARFF_Feature *Data_File_ARFF::createAttribute(char *info) {
		info = trim(info);
		if (*info == 0) {cout << "empty definition!" << endl;return NULL;}

		//  get the second part
		char *val = info+1; while(*val!=' ' && *val!=0 && *val!=9) val++; val[0] = 0; val++;	
		val = trim(val);
		// info => name
		// val => type

		int type = 0;
		int size = 0;

		Data_File_ARFF_Feature *feature = NULL;

		if (val[0]==0)  {cout << "empty datatype!" << endl;return NULL;}
		if (val[0]=='{') {		///  Nominal type
			val++;
			char *p = val;
			vector<string> values;
			while (*val != '}' && *val !=0) {
				if (*val == ',') {
					*val = 0;
//--S [] 2013/03/21: Sang-Wook Lee
#if defined(_MSC_VER)
					std::vector<char> tmp_vec(strlen(p)+1);
					char *tmp = &tmp_vec[0];
#else
//--E [] 2013/03/21: Sang-Wook Lee
					char tmp[strlen(p)+1];
//--S [] 2013/03/21: Sang-Wook Lee
#endif
//--E [] 2013/03/21: Sang-Wook Lee
					strcpy(tmp, p);
					values.push_back((string)trim(tmp));
					p = ++val;
				}
				val++;
			}
			if (*val == 0) {cout << "syntax error! - " << info << endl;return NULL;}
			*val = 0;
//--S [] 2013/03/21: Sang-Wook Lee
#if defined(_MSC_VER)
			std::vector<char> tmp_vec(strlen(p)+1);
			char *tmp = &tmp_vec[0];
#else
//--E [] 2013/03/21: Sang-Wook Lee
			char tmp[strlen(p)+1];
//--S [] 2013/03/21: Sang-Wook Lee
#endif
//--E [] 2013/03/21: Sang-Wook Lee
			strcpy(tmp, p);
			values.push_back((string)trim(tmp));

			feature = new Data_File_ARFF_Feature();
			feature->name = info;
			feature->size = sizeof(short);	//  short
			feature->type = Data_File_ARFF_Feature::TYPE_NOMINAL;
			feature->values = values;

		} else {
			val = mystrupr(val);
			if (strcmp(val, "REAL")==0 ||  strcmp(val, "NUMERIC")==0 || strcmp(val, "INTEGER")==0) {
				type = Data_File_ARFF_Feature::TYPE_NUMERIC;
				size = sizeof(float);
			} else if (strcmp(val, "STRING")==0) {
				type = Data_File_ARFF_Feature::TYPE_STRING;
				size = sizeof(string);
			} else {
				cout << "unknown attribute type : "<< info << " - "<< val << endl;return NULL;
			}
			feature = new Data_File_ARFF_Feature();
			feature->name = info;
			feature->size = size;
			feature->type = type;
		}
		return feature;
	}

	bool Data_File_ARFF::readNextRow(char *row) {
		if (! *row) return false;		//  empty row;
		row = trim(row);
		//if (*row == '{') { cout << "Sparse format not supported yet" << endl; return false;};
		Data_File_ARFF_Record *rec = NULL;
		if (*row == '{') { 
			rec = this->readSparseRow(row);
		} else {
			rec = this->readNormalRow(row);
		}

		if (rec == NULL) return false;
		
		if (rec->getClassId()>=0) classSize[rec->getClassId()] ++;
		this->data.push_back(rec);
		return true;
	}

	bool Data_File_ARFF::readAttrValue(Data_File_ARFF_Feature *feature, Data_File_ARFF_Record *rec, char *value) {
		//  p points to single value for my feature
		char *fdata = (char *)rec->getData()+feature->offset;
		if (feature->type == Data_File_ARFF_Feature::TYPE_NUMERIC) {
			if (sscanf(value, "%f", (float *)fdata)!=1) {
				cout << "Error reading feature data - feature " << feature->name << ":"<< value << endl;return false;
			}
			if (feature->isClass) rec->setClassId((int)*((double*)fdata));
		} else if (feature->type == Data_File_ARFF_Feature::TYPE_STRING) {
		} else if (feature->type == Data_File_ARFF_Feature::TYPE_NOMINAL) {
			vector<string> vals = feature->values;
			int val = -1;
			for (unsigned j=0; j < vals.size(); j++) 
				if (vals[j].compare(value)==0) {val = j; break;};
			if (val == -1) {
				cout << "Nominal value not found - " << feature->name.c_str() << " " << value << endl; return false;
			};
			*(short *)(fdata) = (short)val;

			if (feature->isClass) rec->setClassId(val);
		} else {
			cout << "Unknown attribute type - " << feature->name.c_str() << feature->type << endl; return false;
		}
		return true;
	}

	Data_File_ARFF_Record *Data_File_ARFF::readNormalRow(char *row) {
		Data_File_ARFF_Record *rec = new Data_File_ARFF_Record(recordSize);
		//void *data = rec->getData();
		bool ok = true;

		for (unsigned fi = 0; ok && fi < attributes.size(); fi ++ ) {
			Data_File_ARFF_Feature *feature = attributes[fi];
			if (*row == 0) {	//  problem, row ended to soon
				cout << "Error reading data record - feature " << fi << endl; ok = false; break;
			}
			char *p = row; while (*row != 0 && *row != ',') row++; char tmpch = *row; *row = 0;

			ok = ok && this->readAttrValue(feature, rec, p);

			*row = tmpch;
			if (*row == ',') row++;
		}

		if (! ok ) {delete rec; return NULL;};
		return rec;
	}

	Data_File_ARFF_Record *Data_File_ARFF::readSparseRow(char *row) {

		Data_File_ARFF_Record *rec = new Data_File_ARFF_Record(recordSize);
		void *data = rec->getData();

		memset(data, 0, recordSize);
		bool ok = true;

		//char inString = 0;
		//bool readingID = true;

		char *id;
		char *val;
		int attID = 0;

		row++;
		while (*row != 0 && *row != '}' && ok) {
			//  skip initial spaces
			while (*row == 9 || *row == ' ') row++;
			// start reading attrnum
			id = row;	
			while (*row != 9 && *row != ' ' && *row!=0) row++;
			if (*row == 0) {ok= false; break;};	//  line can't end here
			*row = 0; row++;
			// => id contains attr id

			//  skip spaces
			while (*row == 9 || *row == ' ') row++;
			val = row;
			if (*row == '"' || *row == '\'') { //  value is in commas
				row++;
				while (*row != 0 && *row != *val) *row ++;
				if (*row == 0) {ok= false; break;};	//  line can't end here

				*row = 0; val++; row++;
				while (*row == 9 || *row == ' ' || *row == ',') row++;
			} else {
				while (*row != ',' && *row != '}' && *row != 0) row++;
				if (*row == 0) {ok= false; break;};	//  line can't end here
				if (*row == '}') {*row=0; row++; *row = '}';}
				else {*row = 0; row++;};
				val = trim(val);
			}
			// => val contains attr value
			//	  row contains either ',' or '}'

			attID = atoi(id);
			if (attID<0 || attID >= (int)attributes.size()) {
				cout << "Invalid attribute index - " << attID << endl; ok = false; break;
			}

			Data_File_ARFF_Feature *feature = attributes[attID];
			
			ok = ok && readAttrValue(feature, rec, val);

			while (*row == 9 || *row == ' ') row++;
		}

		if (!ok) {delete rec; return NULL;}

		return rec;
	}

	unsigned int Data_File_ARFF::getNoOfClasses() {
		if (!loaded) {throw fst_error("File not loaded");};

		if (classAttribute == -1) {cout << "class attribute not found" << endl; return 0;};
		if (attributes[classAttribute]->type != Data_File_ARFF_Feature::TYPE_NOMINAL) {
			cout << "Class attribute is not nominal" << endl; return -1;
		}
		return attributes[classAttribute]->values.size();
	}

	unsigned int Data_File_ARFF::getTotalSize() {
		if (!loaded) {throw fst_error("File not loaded");};
		return data.size();
	}

	unsigned int Data_File_ARFF::getClassSize(int clsId) {
		if (!loaded) {throw fst_error("File not loaded");};
		if (classSize == NULL || clsId < 0 || clsId >= classCount) { return -1;};
		return classSize[clsId];
	}

	string Data_File_ARFF::getClassName(int clsId) {
		if (!loaded) {throw fst_error("File not loaded");};
		if (classSize == NULL || clsId < 0 || clsId >= classCount) { return "-n/a-";};
		return attributes[classAttribute]->values[clsId];
	}


	void Data_File_ARFF::sortRecordsByClass() {
		if (!loaded) {throw fst_error("File not loaded");};
		if (classCount < 1) {throw fst_error("Classes not identified");};
		vector<Data_File_ARFF_Record*> **clsData = new vector<Data_File_ARFF_Record*>*[classCount];

		for (int c=0;c<classCount;c++) {
			clsData[c]= new vector<Data_File_ARFF_Record*>;
		}

		for (unsigned i=0;i<data.size();i++) {
			Data_File_ARFF_Record *rec = data[i];
			clsData[rec->getClassId()]->push_back(rec);
		}

		data.clear();
		for (int c=0;c<classCount;c++) {
			vector<Data_File_ARFF_Record*> *v =clsData[c];
			for (unsigned d=0;d<v->size();d++) {
				Data_File_ARFF_Record *rec = v->at(d);
				data.push_back(rec);
			}
			v->clear();
			delete v;
		}

		delete clsData;
	}

}

#endif
