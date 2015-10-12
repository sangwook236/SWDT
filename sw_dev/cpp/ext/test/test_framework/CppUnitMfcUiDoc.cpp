// CppUnitMfcUiDoc.cpp : implementation of the CCppUnitMfcUiDoc class
//

#include "stdafx.h"
#include "CppUnitMfcUi.h"

#include "CppUnitMfcUiDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiDoc

IMPLEMENT_DYNCREATE(CCppUnitMfcUiDoc, CDocument)

BEGIN_MESSAGE_MAP(CCppUnitMfcUiDoc, CDocument)
	//{{AFX_MSG_MAP(CCppUnitMfcUiDoc)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiDoc construction/destruction

CCppUnitMfcUiDoc::CCppUnitMfcUiDoc()
{
	// TODO: add one-time construction code here

}

CCppUnitMfcUiDoc::~CCppUnitMfcUiDoc()
{
}

BOOL CCppUnitMfcUiDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiDoc serialization

void CCppUnitMfcUiDoc::Serialize(CArchive& ar)
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
// CCppUnitMfcUiDoc diagnostics

#ifdef _DEBUG
void CCppUnitMfcUiDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CCppUnitMfcUiDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiDoc commands
