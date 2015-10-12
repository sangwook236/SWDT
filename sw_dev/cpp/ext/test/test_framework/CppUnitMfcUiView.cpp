// CppUnitMfcUiView.cpp : implementation of the CCppUnitMfcUiView class
//

#include "stdafx.h"
#include "CppUnitMfcUi.h"

#include "CppUnitMfcUiDoc.h"
#include "CppUnitMfcUiView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView

IMPLEMENT_DYNCREATE(CCppUnitMfcUiView, CView)

BEGIN_MESSAGE_MAP(CCppUnitMfcUiView, CView)
	//{{AFX_MSG_MAP(CCppUnitMfcUiView)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code!
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView construction/destruction

CCppUnitMfcUiView::CCppUnitMfcUiView()
{
	// TODO: add construction code here

}

CCppUnitMfcUiView::~CCppUnitMfcUiView()
{
}

BOOL CCppUnitMfcUiView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView drawing

void CCppUnitMfcUiView::OnDraw(CDC* pDC)
{
	CCppUnitMfcUiDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	// TODO: add draw code for native data here
}

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView printing

BOOL CCppUnitMfcUiView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CCppUnitMfcUiView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CCppUnitMfcUiView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView diagnostics

#ifdef _DEBUG
void CCppUnitMfcUiView::AssertValid() const
{
	CView::AssertValid();
}

void CCppUnitMfcUiView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CCppUnitMfcUiDoc* CCppUnitMfcUiView::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CCppUnitMfcUiDoc)));
	return (CCppUnitMfcUiDoc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CCppUnitMfcUiView message handlers
