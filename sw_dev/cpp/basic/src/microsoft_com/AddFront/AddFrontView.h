// AddFrontView.h : CAddFrontView 클래스의 인터페이스
//


#pragma once
#include "afxwin.h"
#include "EventHandler.h"

class CAddFrontView : public CFormView
{
protected: // serialization에서만 만들어집니다.
	CAddFrontView();
	DECLARE_DYNCREATE(CAddFrontView)

public:
	enum{ IDD = IDD_ADDFRONT_FORM };

// 특성입니다.
public:
	CAddFrontDoc* GetDocument() const;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.
	virtual void OnInitialUpdate(); // 생성 후 처음 호출되었습니다.

// 구현입니다.
public:
	virtual ~CAddFrontView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonAdd();
	afx_msg void OnBnClickedButtonAddTen();
	afx_msg void OnBnClickedButtonClear();

private:
	int m_iSum;
	int m_iAddEnd;
	IAddBackPtr addBackPtr_;
	CComObject<CEventHandler>* handlerPtr_;
	DWORD cookie_;
public:
	virtual BOOL DestroyWindow();
};

#ifndef _DEBUG  // AddFrontView.cpp의 디버그 버전
inline CAddFrontDoc* CAddFrontView::GetDocument() const
   { return reinterpret_cast<CAddFrontDoc*>(m_pDocument); }
#endif

