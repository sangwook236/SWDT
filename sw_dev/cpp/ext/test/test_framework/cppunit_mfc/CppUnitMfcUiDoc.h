// CppUnitMfcUiDoc.h : interface of the CCppUnitMfcUiDoc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_CPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_)
#define AFX_CPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CCppUnitMfcUiDoc : public CDocument
{
protected: // create from serialization only
	CCppUnitMfcUiDoc();
	DECLARE_DYNCREATE(CCppUnitMfcUiDoc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CCppUnitMfcUiDoc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CCppUnitMfcUiDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CCppUnitMfcUiDoc)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CPPUNITMFCUIDOC_H__981DA87E_7B22_4BF4_9844_5E7CAB60D2F6__INCLUDED_)
