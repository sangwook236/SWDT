/*
 * Copyright 2002,2004 The Apache Software Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Id: DOMPrintFilter.cpp 176026 2004-09-08 13:57:07Z peiyongz $
 * $Log$
 * Revision 1.4  2004/09/08 13:55:31  peiyongz
 * Apache License Version 2.0
 *
 * Revision 1.3  2002/06/04 14:22:51  peiyongz
 * Implement setter/getter from DOMWriterFilter
 *
 * Revision 1.2  2002/06/03 22:40:07  peiyongz
 * *** empty log message ***
 *
 * Revision 1.1  2002/05/29 13:33:32  peiyongz
 * DOM3 Save Interface: DOMWriter/DOMWriterFilter
 *
 */

#include "xercesc_DOMPrintFilter.hpp"
#include <xercesc/util/XMLUniDefs.hpp>
#include <xercesc/util/XMLString.hpp>


static const XMLCh element_person[]=
{
	XERCES_CPP_NAMESPACE::chLatin_p, XERCES_CPP_NAMESPACE::chLatin_e, XERCES_CPP_NAMESPACE::chLatin_r, XERCES_CPP_NAMESPACE::chLatin_s, XERCES_CPP_NAMESPACE::chLatin_o, XERCES_CPP_NAMESPACE::chLatin_n, XERCES_CPP_NAMESPACE::chNull
};

static const XMLCh element_link[]=
{
	XERCES_CPP_NAMESPACE::chLatin_l, XERCES_CPP_NAMESPACE::chLatin_i, XERCES_CPP_NAMESPACE::chLatin_n, XERCES_CPP_NAMESPACE::chLatin_k, XERCES_CPP_NAMESPACE::chNull
};

DOMPrintFilter::DOMPrintFilter(ShowType whatToShow)
: fWhatToShow(whatToShow)
{}

XERCES_CPP_NAMESPACE::DOMNodeFilter::FilterAction DOMPrintFilter::acceptNode(const XERCES_CPP_NAMESPACE::DOMNode *node) const
{
	//
	// The DOMWriter shall call getWhatToShow() before calling acceptNode(), to show nodes which are supposed to be shown to this filter.
	//
	// REVISIT: In case the DOMWriter does not follow the protocol,
	//          Shall the filter honour, or NOT, what it claimes
	//         (when it is constructed/setWhatToShow())
	//          it is interested in ?
	//
	// The DOMLS specs does not specify that acceptNode() shall do this way, or not, so it is up the implementation,
	// to skip the code below for the sake of performance ...
	//
	if ((getWhatToShow() & (1 << (node->getNodeType() - 1))) == 0)
		return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_ACCEPT;

	switch (node->getNodeType())
	{
	case XERCES_CPP_NAMESPACE::DOMNode::ELEMENT_NODE:
		{
			// for element whose name is "person", skip it
			if (XERCES_CPP_NAMESPACE::XMLString::compareString(node->getNodeName(), element_person) == 0)
				return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_SKIP;
			// for element whose name is "line", reject it
			if (XERCES_CPP_NAMESPACE::XMLString::compareString(node->getNodeName(), element_link) == 0)
				return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_REJECT;
			// for rest, accept it
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_ACCEPT;

			break;
		}
	case XERCES_CPP_NAMESPACE::DOMNode::COMMENT_NODE:
		{
			// the WhatToShow will make this no effect
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_REJECT;
			break;
		}
	case XERCES_CPP_NAMESPACE::DOMNode::TEXT_NODE:
		{
			// the WhatToShow will make this no effect
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_REJECT;
			break;
		}
	case XERCES_CPP_NAMESPACE::DOMNode::DOCUMENT_TYPE_NODE:
		{
			// even we say we are going to process document type, we are not able be to see this node since
			// DOMWriterImpl (a XercesC's default implementation of DOMWriter) will not pass DocumentType node to this filter.
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_REJECT;  // no effect
			break;
		}
	case XERCES_CPP_NAMESPACE::DOMNode::DOCUMENT_NODE:
		{
			// same as DOCUMENT_NODE
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_REJECT;  // no effect
			break;
		}
	default:
		{
			return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_ACCEPT;
			break;
		}
	}

	return XERCES_CPP_NAMESPACE::DOMNodeFilter::FILTER_ACCEPT;
}

