// TestCppUnitMfcUiView.cpp : implementation of the CTestCppUnitMfcUiView class
//

#include "stdafx.h"
#include "TestCppUnitMfcUi.h"

#include "TestCppUnitMfcUiDoc.h"
#include "TestCppUnitMfcUiView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView

IMPLEMENT_DYNCREATE(CTestCppUnitMfcUiView, CView)

BEGIN_MESSAGE_MAP(CTestCppUnitMfcUiView, CView)
	//{{AFX_MSG_MAP(CTestCppUnitMfcUiView)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView construction/destruction

CTestCppUnitMfcUiView::CTestCppUnitMfcUiView()
{
	// TODO: add construction code here

}

CTestCppUnitMfcUiView::~CTestCppUnitMfcUiView()
{
}

BOOL CTestCppUnitMfcUiView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView drawing

void CTestCppUnitMfcUiView::OnDraw(CDC* pDC)
{
	CTestCppUnitMfcUiDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	// TODO: add draw code for native data here
}

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView printing

BOOL CTestCppUnitMfcUiView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CTestCppUnitMfcUiView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CTestCppUnitMfcUiView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView diagnostics

#ifdef _DEBUG
void CTestCppUnitMfcUiView::AssertValid() const
{
	CView::AssertValid();
}

void CTestCppUnitMfcUiView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CTestCppUnitMfcUiDoc* CTestCppUnitMfcUiView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CTestCppUnitMfcUiDoc)));
	return (CTestCppUnitMfcUiDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CTestCppUnitMfcUiView message handlers
