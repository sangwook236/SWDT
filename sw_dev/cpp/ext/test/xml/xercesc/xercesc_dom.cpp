//#include "stdafx.h"
#include "xercesc_DOMTreeErrorReporter.hpp"
#include "xercesc_DOMPrintFilter.hpp"
#include "xercesc_DOMPrintErrorHandler.hpp"
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
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


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_xercesc {

int dom()
{
	try
	{
		XERCES_CPP_NAMESPACE::XMLPlatformUtils::Initialize();
	}
	catch (const XERCES_CPP_NAMESPACE::XMLException& e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Error during initialization! :" << std::endl
			<< e.getMessage() << std::endl;
#else
		const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
		std::cout << "Error during initialization! :" << std::endl
			<< message << std::endl;
		XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
		return 1;
	}

	XERCES_CPP_NAMESPACE::XercesDOMParser *parser = new XERCES_CPP_NAMESPACE::XercesDOMParser();
	// Indicates what validation scheme to use. It defaults to 'auto', but can be set via the -v= command.
	parser->setValidationScheme(XERCES_CPP_NAMESPACE::XercesDOMParser::Val_Always);
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
    // FIXME [check] >> is it correct? convert 'wchar_t *' to 'XMLCh *'
	//const XMLCh *xmlFile = XMLStrL("./data/xml/xerces/personal.xml");
	//const XMLCh *xmlFile = (XMLCh *)L"./data/xml/xerces/personal.xml";
	const XMLCh *xmlFile = (XMLCh *)L"./data/xml/xerces/books.xml";
#else
	//const char *xmlFile = "./data/xml/xerces/personal.xml";
	const char *xmlFile = "./data/xml/xerces/books.xml";
#endif

	try
	{
		parser->parse(xmlFile);
	}
	catch (const XERCES_CPP_NAMESPACE::OutOfMemoryException &)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"OutOfMemoryException" << std::endl;
#else
		std::cerr << "OutOfMemoryException" << std::endl;
#endif
		return -1;
	}
	catch (const XERCES_CPP_NAMESPACE::XMLException &e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Exception message is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
		std::cout << "Exception message is: " << std::endl
			<< message << std::endl;
		XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
		return -1;
	}
	catch (const XERCES_CPP_NAMESPACE::DOMException &e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Exception message is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
		std::cout << "Exception message is: " << std::endl
			<< message << std::endl;
		XERCES_CPP_NAMESPACE::XMLString::release(&message);
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
#if defined(_UNICODE) || defined(UNICODE)
	const XMLCh *outputEncoding = (XMLCh *)L"UTF-8";
#else
	const XMLCh *outputEncoding = "UTF-8";
#endif
	try
	{
		// get a serializer, an instance of DOMWriter
#if defined(_UNICODE) || defined(UNICODE)
        // FIXME [check] >> is it correct? convert 'wchar_t *' to 'XMLCh *'
		XERCES_CPP_NAMESPACE::DOMImplementation *impl = XERCES_CPP_NAMESPACE::DOMImplementationRegistry::getDOMImplementation((XMLCh *)L"LS");
#else
		XERCES_CPP_NAMESPACE::DOMImplementation *impl = XERCES_CPP_NAMESPACE::DOMImplementationRegistry::getDOMImplementation("LS");
#endif
		XERCES_CPP_NAMESPACE::DOMLSSerializer *theSerializer = ((XERCES_CPP_NAMESPACE::DOMImplementationLS *)impl)->createLSSerializer();
        XERCES_CPP_NAMESPACE::DOMLSOutput *theOutputDesc = ((XERCES_CPP_NAMESPACE::DOMImplementationLS *)impl)->createLSOutput();

		// set user specified output encoding
		// FIXME [check] >> is it correct? convert 'wchar *' to 'XMLCh *'
		theOutputDesc->setEncoding(outputEncoding);

		// plug in user's own filter
		if (useFilter)
		{
			// even we say to show attribute, but the DOMWriter will not show attribute nodes to the filter as
			// the specs explicitly says that DOMWriter shall NOT show attributes to DOMWriterFilter.
			//
			// so DOMNodeFilter::SHOW_ATTRIBUTE has no effect.
			// same DOMNodeFilter::SHOW_DOCUMENT_TYPE, no effect.
			myFilter = new DOMPrintFilter(
                XERCES_CPP_NAMESPACE::DOMNodeFilter::SHOW_ELEMENT |
				XERCES_CPP_NAMESPACE::DOMNodeFilter::SHOW_ATTRIBUTE |
				XERCES_CPP_NAMESPACE::DOMNodeFilter::SHOW_DOCUMENT_TYPE
            );
			theSerializer->setFilter(myFilter);
		}

		// plug in user's own error handler
		XERCES_CPP_NAMESPACE::DOMErrorHandler *myErrorHandler = new DOMPrintErrorHandler();
        XERCES_CPP_NAMESPACE::DOMConfiguration *serializerConfig = theSerializer->getDomConfig();
		serializerConfig->setParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMErrorHandler, myErrorHandler);

		// set feature if the serializer supports the feature/mode

		// Indicates whether split-cdata-sections is to be enabled or not.
		const bool splitCdataSections = true;
		if (serializerConfig->canSetParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTSplitCdataSections, splitCdataSections))
			serializerConfig->setParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTSplitCdataSections, splitCdataSections);

		// Indicates whether default content is discarded or not.
		const bool discardDefaultContent = true;
		if (serializerConfig->canSetParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTDiscardDefaultContent, discardDefaultContent))
			serializerConfig->setParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTDiscardDefaultContent, discardDefaultContent);

		const bool formatPrettyPrint = false;
		if (serializerConfig->canSetParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTFormatPrettyPrint, formatPrettyPrint))
			serializerConfig->setParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTFormatPrettyPrint, formatPrettyPrint);

		const bool writeBOM = false;
		if (serializerConfig->canSetParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTBOM, writeBOM))
			serializerConfig->setParameter(XERCES_CPP_NAMESPACE::XMLUni::fgDOMWRTBOM, writeBOM);

		// Plug in a format target to receive the resultant XML stream from the serializer.
		// StdOutFormatTarget prints the resultant XML stream to stdout once it receives any thing from the serializer.
		XERCES_CPP_NAMESPACE::XMLFormatTarget *myFormTarget = NULL;
		const char *outputFile = NULL;
		if (outputFile)
			myFormTarget = new XERCES_CPP_NAMESPACE::LocalFileFormatTarget(outputFile);
		else
			myFormTarget = new XERCES_CPP_NAMESPACE::StdOutFormatTarget();
        theOutputDesc->setByteStream(myFormTarget);

		// get the DOM representation
        XERCES_CPP_NAMESPACE::DOMDocument *doc = parser->getDocument();

        // do the serialization through DOMLSSerializer::write();
		//const char *xPathExpression = NULL;
		//const char *xPathExpression = "//*";
		//const char *xPathExpression = "//book";
		//const char *xPathExpression = "//title";
		const char *xPathExpression = "//author";
		//const char *xPathExpression = "//@lang";
		//const char *xPathExpression = "/bookstore/book/title";  // selects all the title nodes
		//const char *xPathExpression = "/bookstore/book[1]/title";  // selects the title of the first book node under the bookstore element
		//const char *xPathExpression = "/bookstore/book/price/text()";  // selects the text from all the price nodes
		//const char *xPathExpression = "/bookstore/book[price>35]/price";  // selects all the price nodes with a price higher than 35
		//const char *xPathExpression = "/bookstore/book[price>35]/title";  // selects all the title nodes with a price higher than 35
        if (NULL != xPathExpression)
        {
            XMLCh *xpathStr = XERCES_CPP_NAMESPACE::XMLString::transcode(xPathExpression);
            XERCES_CPP_NAMESPACE::DOMElement *root = doc->getDocumentElement();
            try
            {
                XERCES_CPP_NAMESPACE::DOMXPathNSResolver *resolver = doc->createNSResolver(root);
                XERCES_CPP_NAMESPACE::DOMXPathResult *result = doc->evaluate(
                    xpathStr,
                    root,
                    resolver,
                    XERCES_CPP_NAMESPACE::DOMXPathResult::ORDERED_NODE_SNAPSHOT_TYPE,
                    NULL
				);

                XMLSize_t nLength = result->getSnapshotLength();
                for (XMLSize_t i = 0; i < nLength; ++i)
                {
                    result->snapshotItem(i);
                    theSerializer->write(result->getNodeValue(), theOutputDesc);
                }

                result->release();
                resolver->release ();
            }
            catch (const XERCES_CPP_NAMESPACE::DOMXPathException &e)
            {
#if defined(_UNICODE) || defined(UNICODE)
				std::wcerr << L"An error occurred during processing of the XPath expression. Msg is:: " << std::endl
					<< e.getMessage() << std::endl;
#else
				const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
				std::cerr << "An error occurred during processing of the XPath expression. Msg is: " << std::endl
					<< message << std::endl;
				XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
            }
            catch(const XERCES_CPP_NAMESPACE::DOMException &e)
            {
#if defined(_UNICODE) || defined(UNICODE)
				std::wcerr << L"An error occurred during processing of the XPath expression. Msg is:: " << std::endl
					<< e.getMessage() << std::endl;
#else
				const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
				std::cerr << "An error occurred during processing of the XPath expression. Msg is: " << std::endl
					<< message << std::endl;
				XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
            }
            XERCES_CPP_NAMESPACE::XMLString::release(&xpathStr);
        }
        else
            theSerializer->write(doc, theOutputDesc);

        theOutputDesc->release();
        theSerializer->release();

		// Filter, formatTarget and error handler are NOT owned by the serializer.
		delete myFormTarget;
		delete myErrorHandler;

		if (useFilter)
			delete myFilter;
	}
	catch (const XERCES_CPP_NAMESPACE::OutOfMemoryException &)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"OutOfMemoryException" << std::endl;
#else
		std::cerr << "OutOfMemoryException" << std::endl;
#endif
	}
	catch (XERCES_CPP_NAMESPACE::XMLException &e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcerr << L"An error occurred during creation of output transcoder. Msg is: " << std::endl
			<< e.getMessage() << std::endl;
#else
		const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
		std::cerr << "An error occurred during creation of output transcoder. Msg is: " << std::endl
			<< message << std::endl;
		XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
	}

	delete errReporter;
	delete parser;

    //XERCES_CPP_NAMESPACE::XMLString::release(&outputEncoding);

	try
	{
		XERCES_CPP_NAMESPACE::XMLPlatformUtils::Terminate();
	}
	catch (const XERCES_CPP_NAMESPACE::XMLException &e)
	{
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"Error during Termination! :" << std::endl
			<< e.getMessage() << std::endl;
#else
		const char *message = XERCES_CPP_NAMESPACE::XMLString::transcode(e.getMessage());
		std::cout << "Error during Termination! :" << std::endl
			<< message << std::endl;
		XERCES_CPP_NAMESPACE::XMLString::release(&message);
#endif
		return 1;
	}

	return 0;
}

}  // namespace my_xercesc
