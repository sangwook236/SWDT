///
/// \file   DataTable.cpp
/// \brief  Implementation of stream output for errors from DataTable class
/// \author Kent Holsinger
/// \date   2005-05-18
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

// local includes
#include "mcmc++/DataTable.h"

/// Stream output for DataTable errors
///
/// \param out     The stream for output
/// \param result  The error identifier
///
std::ostream& operator<< (std::ostream& out, 
                          enum DataTableResult result) 
{
  std::string s;
  switch (result) {
    case readSuccess:
      s = "Success";
      break;
    case labelError:
      s = "Label error";
      break;
    case valueError:
      s = "Value error";
      break;
    case openError:
      s = "Open error";
      break;
    case grammarError:
      s = "Parsing error\n";
      break;
    default:
      s = "Unrecognized internal error";
      break;
  }
  return out << s;
}


