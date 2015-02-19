///
/// \file   DataTable.h
/// \brief  Provides DataTable class for access to tabular data
///
/// The DataTable class reads homogeneous tabular data, i.e., numerical data 
/// that is either all of the same type or that can be converted to the base
/// type of the data table using standard conversions. Rows or columns or both
/// can be labeled, but labels are not required.
///
/// \author Kent Holsinger
/// \date   2004-06-26
///

// This file is part of MCMC++, a library for constructing C++ programs
// that implement MCMC analyses of Bayesian statistical models.
// Copyright (c) 2004-2006 Kent E. Holsinger
//
// MCMC++ is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// MCMC++ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with MCMC++; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#if !defined(__DATATABLE_H)
#define __DATATABLE_H

// standard includes
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
// boost includes
#include <boost/tokenizer.hpp>
//--S [] 2015/02/15 : Sang-Wook Lee
//#include <boost/spirit/core.hpp>
//#include <boost/spirit/utility.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/spirit/include/classic_utility.hpp>
//--E [] 2015/02/15 : Sang-Wook Lee
// local includes
#include "mcmc++/util.h"

/// enum DataTableResult
///
/// codes used to determine whether read was successful and the type
/// of error, if not
///
enum DataTableResult {
  readSuccess = 0,
  labelError,
  valueError,
  openError,
  notEmptyError,
  grammarError
};

template <typename Type>
class DataTableGrammar;

/// \class BadCol
/// \brief Exception thrown on bad column index
///
class BadCol {};

/// \class BadRow
/// \brief Exception thrown on bad row index
///
class BadRow {};

/// argCheck_ controls whether row and column indexes are bounds checked
/// before use
///
/// Defaults to 1 (true) unles NDEBUG is defined
///
#if defined(NDEBUG)
#define argCheck_ 0
#else
#define argCheck_ 1
#endif


/// \class DataTable
/// \brief Provides access to homogeneous tabular data
///
/// The DataTable class reads homogeneous tabular data, i.e., numerical data 
/// that is either all of type Type or that can be converted to Type using
/// standard conversions. Rows or columns or both can be labeled, but labels 
/// are not required. A simple method for stream output of errors is also 
/// provided.
///
template <typename Type>
class DataTable {
  enum {
    defaultWidth = 14,      ///< default width of field in output
    defaultColumnSpace = 2  ///< default number of spaces between output columns
  };

public:
  /// Constructor -- no default constructor is provided.
  ///
  /// \param columnLabels   Are columns labeled?
  /// \param rowLabels      Are rows labeled?
  ///
  /// Initializes data structures. Use Read() to collect the data.
  ///
  DataTable(const bool columnLabels = true, const bool rowLabels = false)
    : width_(defaultWidth), nRows_(0), nCols_(0), columnLabels_(columnLabels), 
      rowLabels_(rowLabels)
  {}

  /// Read data from a file.
  ///
  /// \param fileName       The name of the file from which data is to be read
  /// \return notEmptyError If the DataTable is not empty
  /// \return labelError    If there is an error reading column labels
  /// \return valueError    If there is an error reading values
  /// \return openError     If filename could not be opened for reading
  /// \return readSuccess   If everything works
  ///
  /// The DataTable must be empty for data to be read. If it has been used
  /// before, Flush() must be used to re-initialize the internal state.
  enum DataTableResult Read(const std::string fileName) {
    enum DataTableResult result = readSuccess; // readSuccess == 0
    if ((nRows_ > 0) || (nCols_ > 0)) { // explicit flush required
      result = notEmptyError;
    }
    std::ifstream input(fileName.c_str());
    if (input) {
      if (columnLabels_ && !ReadLabels(input)) {
        result = labelError;
      } else if (!ReadValues(input)) {
        result = valueError;
      }
      input.close();
    } else {
      result = openError;
    }
    return result;
  }

  /// Sets width of output based on length of string
  ///
  /// \param s   The string used to set the width
  ///
  void SetWidth(const std::string s) {
    if (boost::is_integral<Type>::value) {
      width_ = 4;
    } else {
      width_ = std::max(width_, s.length() + defaultColumnSpace);
    }
  }

  /// Value of the data at specified row and column
  ///
  /// \param row   Index of the data row
  /// \param col   Index of the data column
  ///
  inline Type Value(const unsigned row, const unsigned col) const {
    Util::Assert<BadRow>(!argCheck_ || ((row < nRows_) && (row >= 0)));
    Util::Assert<BadCol>(!argCheck_ || ((col < nCols_) && (col >= 0)));
    return data_[row][col];
  }

  /// Set value of the data at specified row and column
  ///
  /// \param row   Index of the data row
  /// \param col   Index of the data column
  /// \param value Value to be inserted
  ///
  inline void SetValue(const unsigned row, const unsigned col,
                       const Type value)
  {
    Util::Assert<BadRow>(!argCheck_ || ((row < nRows_) && (row >= 0)));
    Util::Assert<BadCol>(!argCheck_ || ((col < nCols_) && (col >= 0)));
    data_[row][col] = value;
  }

  /// Label associated with a particular column index 
  ///
  /// \param index   column index
  ///
  inline std::string ColumnLabel(const unsigned index) const {
    Util::Assert<BadCol>(!argCheck_ 
                         || ((index < nLabelCols_) && (index >= 0)));
    return cLabels_.at(index);
  }

  /// Label associated with a particular row index 
  ///
  /// \param index   row index
  ///
  inline std::string RowLabel(const unsigned index) const {
    Util::Assert<BadRow>(!argCheck_ || ((index < nRows_) && (index >= 0)));
    return rLabels_.at(index);
  }

  /// An entire row of the data matrix
  ///
  /// \param row   Index of the data row
  ///
  std::vector<Type> RowVector(const unsigned row) const {
    Util::Assert<BadRow>(!argCheck_ || ((row < nRows_) && (row >= 0)));
    return data_[row];
  }

  /// An entire column of the data matrix
  ///
  /// \param col   Index of the data column
  ///
  std::vector<Type> ColumnVector(const unsigned col) const {
    Util::Assert<BadCol>(!argCheck_ || ((col < nCols_) && (col >= 0)));
    std::vector<Type> x(nRows_);
    for (unsigned i = 0; i < nRows_; ++i) {
      x[i] = Value(i, col);
    }
    return x;
  }

  /// Print the table to the specified stream 
  ///
  /// \param out   The stream for output (defaults to std::cout)
  ///
  void PrintTable(std::ostream& out = std::cout) {
    if (columnLabels_) {
      PrintLabels(out);
    }
    for (unsigned i = 0; i < nRows_; ++i) {
      if (rowLabels_) {
        out << RowLabel(i) << ": ";
      }
      PrintValueRow(out, i);
    }
  }

  /// Re-initialize internal data structures.
  ///
  void Flush(void) {
    cLabels_.clear();
    rLabels_.clear();
    data_.clear();
    nRows_ = nCols_ = nLabelCols_ = 0;
  }

  /// Number of rows in the data
  ///
  inline unsigned Rows(void) const {
    return nRows_;
  }

  /// Number of columns in the data
  ///
  inline unsigned Columns(void) const {
    return nCols_;
  }

  /// Number of column labels
  ///
  inline unsigned ColumnLabels(void) const {
    return nLabelCols_;
  }

  /// Set all data elements to zero
  ///
  void SetZero(void) {
    for (unsigned i = 0; i < nRows_; ++i) {
      for (unsigned j = 0; j < nCols_; ++j) {
        data_[i][j] = 0;
      }
    }
  }

private:

  bool ReadLabels(std::istream& in) {
    using namespace boost;
    
    std::string s;
    std::getline(in, s);
    typedef tokenizer<char_separator<char> > localTokenizer;
    char_separator<char> sep(" \t:`~!@#$%^&*()+={}[]\\|;:\'\",.<>/?\r");
    localTokenizer tok(s, sep);
    for (localTokenizer::iterator i = tok.begin(); i != tok.end(); ++i) {
      cLabels_.push_back(*i);
    }
    nLabelCols_ = cLabels_.size();
    return nLabelCols_ > 0;
  }

  bool ReadValues(std::istream& in) {
    using namespace boost;
	//--S [] 2015/02/15 : Sang-Wook Lee
	//using namespace boost::spirit;
	using namespace boost::spirit::classic;
	//--E [] 2015/02/15 : Sang-Wook Lee

    std::string s;
    parse_info<> info;
    bool result = true;
    while (std::getline(in, s)) {
      // needed (for some reason) for tables that start without spaces
      s.insert(s.begin(), ' ');
      data_.resize(nRows_ + 1);
      std::vector<double> tempData;
      if (rowLabels_) {
        typedef tokenizer<char_separator<char> > localTokenizer;
        char_separator<char> sep(" \t:`~!@#$%^&*()+={}[]\\|;:\'\",.<>/?");
        localTokenizer tok(s, sep);
        localTokenizer::iterator i = tok.begin();
        rLabels_.push_back(*i);
        chset<> alnum("0-9a-zA-Z");
        info = parse(s.c_str(),
                     //
                     ( 
                      alnum >> *(ch_p('-') | ch_p('_') | alnum) >>
                      ch_p(':') >> 
                      *space_p >> 
                      real_p[append(tempData)] >>
                      *(*space_p >> real_p[append(tempData)]) 
                     ),
                     //
                     space_p);
      } else {
        info = parse(s.c_str(),
                     //
                     (real_p[append(tempData)] >>
                      *(*space_p >> real_p[append(tempData)]) ),
                     //
                     space_p);
      }
      for (unsigned i = 0; i < tempData.size(); ++i) {
        data_[nRows_].push_back(static_cast<Type>(tempData[i]));
      }
      result = (result && info.hit);
      ++nRows_;
    }
    if (result && (nCols_ == 0)) {
      nCols_ = data_[0].size();
    }
    return (result && (nRows_ > 0));
  }
  
  void PrintLabels(std::ostream& out) {
    typedef std::vector<std::string>::const_iterator LabelIter;
    LabelIter end = cLabels_.end();
    for (LabelIter i = cLabels_.begin(); i != end; ++i) {
      SetWidth(*i);
    }
    for (LabelIter i = cLabels_.begin(); i != end; ++i) {
      Print(out, *i);
    }
    out << std::endl;
  }

  void PrintValueRow(std::ostream& out, const unsigned row) {
    Util::Assert<BadRow>(!argCheck_ || ((row < nRows_) && (row >= 0)));
    for (unsigned col = 0; col < nCols_; ++col) {
      out << std::setw(width_) << data_[row][col];
    }
    out << std::endl;
  }

  template <class OutputType>
  void Print(std::ostream& out, const OutputType& s) {
    out << std::setw(width_+2) << s;
  }

  std::vector<std::string> rLabels_;
  std::vector<std::string> cLabels_;
  std::vector<std::vector<Type> > data_;

  unsigned width_;
  unsigned nRows_;
  unsigned nCols_;
  unsigned nLabelCols_;

  bool columnLabels_;
  bool rowLabels_;

};

std::ostream& operator<< (std::ostream& out, 
                          enum DataTableResult result);

#endif

// Local Variables: //
// mode: c++ //
// End: //
