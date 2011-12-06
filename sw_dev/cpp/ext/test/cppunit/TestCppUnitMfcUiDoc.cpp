// TestCppUnitMfcUiDoc.cpp : implementation of the CTestCppUnitMfcUiDoc class
//

#include "stdafx.h"
#include "TestCppUnitMfcUi.h"

#include "TestCppUnitMfcUiDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiDoc

IMPLEMENT_DYNCREATE(CTestCppUnitMfcUiDoc, CDocument)

BEGIN_MESSAGE_MAP(CTestCppUnitMfcUiDoc, CDocument)
	//{{AFX_MSG_MAP(CTestCppUnitMfcUiDoc)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiDoc construction/destruction

CTestCppUnitMfcUiDoc::CTestCppUnitMfcUiDoc()
{
	// TODO: add one-time construction code here

}

CTestCppUnitMfcUiDoc::~CTestCppUnitMfcUiDoc()
{
}

BOOL CTestCppUnitMfcUiDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiDoc serialization

void CTestCppUnitMfcUiDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiDoc diagnostics

#ifdef _DEBUG
void CTestCppUnitMfcUiDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CTestCppUnitMfcUiDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiDoc commands
