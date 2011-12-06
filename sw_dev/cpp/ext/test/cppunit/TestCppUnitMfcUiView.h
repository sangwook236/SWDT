// TestCppUnitMfcUiView.h : interface of the CTestCppUnitMfcUiView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_TESTCPPUNITMFCUIVIEW_H__9522ED9B_3261_4447_905B_AF11E74015DD__INCLUDED_)
#define AFX_TESTCPPUNITMFCUIVIEW_H__9522ED9B_3261_4447_905B_AF11E74015DD__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CTestCppUnitMfcUiView : public CView
{
protected: // create from serialization only
	CTestCppUnitMfcUiView();
	DECLARE_DYNCREATE(CTestCppUnitMfcUiView)

// Attributes
public:
	CTestCppUnitMfcUiDoc* GetDocument();

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTestCppUnitMfcUiView)
	public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CTestCppUnitMfcUiView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CTestCppUnitMfcUiView)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in TestCppUnitMfcUiView.cpp
inline CTestCppUnitMfcUiDoc* CTestCppUnitMfcUiView::GetDocument()
   { return (CTestCppUnitMfcUiDoc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTCPPUNITMFCUIVIEW_H__9522ED9B_3261_4447_905B_AF11E74015DD__INCLUDED_)
