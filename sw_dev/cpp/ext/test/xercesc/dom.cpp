//#include "stdafx.h"
#include "DOMTreeErrorReporter.hpp"
#include "DOMPrintFilter.hpp"
#include "DOMPrintErrorHandler.hpp"
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>

#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif


XERCES_CPP_NAMESPACE_USE

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

int dom()
{
	try
	{
		XMLPlatformUtils::Initialize();
	}
	catch (const XMLException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Error during initialization! :" << std::endl
			<< e.getMessage() << std::endl;
#else
		char* message = XMLString::transcode(e.getMessage());
		std::cout << "Error during initialization! :" << std::endl
			<< message << std::endl;
		XMLString::release(&message);
#endif
		return 1;
	} 

	XercesDOMParser* parser = new XercesDOMParser();
	// Indicates what validation scheme to use. It defaults to 'auto', but can be set via the -v= command.
	parser->setValidationScheme(XercesDOMParser::Val_Always);
	// Indicates whether namespace processing should be done.
	parser->setDoNamespaces(true);    // optional
	// Indicates whether schema processing should be done.
	parser->setDoSchema(true);
	// Indicates whether full schema constraint checking should be done.
	parser->setValidationSchemaFullChecking(true);
	// Indicates whether entity reference nodes needs to be created or not. Defaults to false
	parser->setCreateEntityReferenceNodes(true);

	DOMTreeErrorReporter *errReporter = new DOMTreeErrorReporter();
	parser->setErrorHandler(errReporter);

	//char buf[1000];
	//getcwd(buf, 1000);

#if defined(_UNICODE) || defined(UNICODE)
	const wchar_t* xmlFile = L".\\xerces_data\\personal.xml";
#else
	const char* xmlFile = ".\\xerces_data\\personal.xml";
#endif

	try
	{
		parser->parse(xmlFile);
	}
	catch (const OutOfMemoryException&)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"OutOfMemoryException" << std::endl;
#else
		std::cerr << "OutOfMemoryException" << std::endl;
#endif
		return -1;
	}
	catch (const XMLException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Exception message is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		char* message = XMLString::transcode(e.getMessage());
		std::cout << "Exception message is: " << std::endl
			<< message << std::endl;
		XMLString::release(&message);
#endif
		return -1;
	}
	catch (const DOMException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Exception message is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		char* message = XMLString::transcode(e.getMessage());
		std::cout << "Exception message is: " << std::endl
			<< message << std::endl;
		XMLString::release(&message);
#endif
		return -1;
	}
	catch (...)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Unexpected Exception" << std::endl;
#else
		std::cout << "Unexpected Exception" << std::endl;
#endif
		return -1;
	}

	//
	DOMPrintFilter *myFilter = 0;

	// Indicates if user wants to plug in the DOMPrintFilter.
	const bool useFilter = false;
	try
	{
		// get a serializer, an instance of DOMWriter
#if defined(_UNICODE) || defined(UNICODE)
		DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(L"LS");
#else
		DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation("LS");
#endif
		DOMWriter *theSerializer = ((DOMImplementationLS*)impl)->createDOMWriter();

		// set user specified output encoding
		theSerializer->setEncoding(L"UTF-8");

		// plug in user's own filter
		if (useFilter)
		{
			// even we say to show attribute, but the DOMWriter will not show attribute nodes to the filter as
			// the specs explicitly says that DOMWriter shall NOT show attributes to DOMWriterFilter.
			//
			// so DOMNodeFilter::SHOW_ATTRIBUTE has no effect.
			// same DOMNodeFilter::SHOW_DOCUMENT_TYPE, no effect.
			myFilter = new DOMPrintFilter(DOMNodeFilter::SHOW_ELEMENT |
				DOMNodeFilter::SHOW_ATTRIBUTE |
				DOMNodeFilter::SHOW_DOCUMENT_TYPE);
			theSerializer->setFilter(myFilter);
		}

		// plug in user's own error handler
		DOMErrorHandler *myErrorHandler = new DOMPrintErrorHandler();
		theSerializer->setErrorHandler(myErrorHandler);

		// set feature if the serializer supports the feature/mode

		// Indicates whether split-cdata-sections is to be enabled or not.
		if (theSerializer->canSetFeature(XMLUni::fgDOMWRTSplitCdataSections, true))
			theSerializer->setFeature(XMLUni::fgDOMWRTSplitCdataSections, true);

		// Indicates whether default content is discarded or not.
		if (theSerializer->canSetFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true))
			theSerializer->setFeature(XMLUni::fgDOMWRTDiscardDefaultContent, true);

		if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, false))
			theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, false);

		if (theSerializer->canSetFeature(XMLUni::fgDOMWRTBOM, false))
			theSerializer->setFeature(XMLUni::fgDOMWRTBOM, false);

		// Plug in a format target to receive the resultant XML stream from the serializer.
		// StdOutFormatTarget prints the resultant XML stream to stdout once it receives any thing from the serializer.
		XMLFormatTarget *myFormTarget;
		const char *outputFile = 0L;
		if (outputFile)
			myFormTarget = new LocalFileFormatTarget(outputFile);
		else
			myFormTarget = new StdOutFormatTarget();

		// get the DOM representation
		DOMNode *doc = parser->getDocument();

		// do the serialization through DOMWriter::writeNode();
		theSerializer->writeNode(myFormTarget, *doc);

		delete theSerializer;

		// Filter, formatTarget and error handler are NOT owned by the serializer.
		delete myFormTarget;
		delete myErrorHandler;

		if (useFilter)
			delete myFilter;
	}
	catch (const OutOfMemoryException&)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"OutOfMemoryException" << std::endl;
#else
		std::cerr << "OutOfMemoryException" << std::endl;
#endif
	}
	catch (XMLException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"An error occurred during creation of output transcoder. Msg is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		char *message = XMLString::transcode(e.getMessage());
		std::cerr << "An error occurred during creation of output transcoder. Msg is: " << std::endl
			<< message << std::endl;
		XMLString::release(&message);
#endif
	}

	delete parser;
	delete errReporter;

	try
	{
		XMLPlatformUtils::Terminate();
	}
	catch (const XMLException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Error during Termination! :" << std::endl
			<< e.getMessage() << std::endl;
#else
		char *message = XMLString::transcode(e.getMessage());
		std::cout << "Error during Termination! :" << std::endl
			<< message << std::endl;
		XMLString::release(&message);
#endif
		return 1;
	}

	return 0;
}
