// TestCppUnitMfcUiDoc.h : interface of the CTestCppUnitMfcUiDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_TESTCPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_)
#define AFX_TESTCPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CTestCppUnitMfcUiDoc : public CDocument
{
protected: // create from serialization only
	CTestCppUnitMfcUiDoc();
	DECLARE_DYNCREATE(CTestCppUnitMfcUiDoc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CTestCppUnitMfcUiDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CTestCppUnitMfcUiDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CTestCppUnitMfcUiDoc)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTCPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_)
