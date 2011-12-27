// AddFront_byAttribView.h : CAddFront_byAttribView 클래스의 인터페이스
//


#pragma once
#include "EventHandler.h"

class CAddFront_byAttribView : public CFormView
{
protected: // serialization에서만 만들어집니다.
	CAddFront_byAttribView();
	DECLARE_DYNCREATE(CAddFront_byAttribView)

public:
	enum{ IDD = IDD_ADDFRONT_BYATTRIB_FORM };

// 특성입니다.
public:
	CAddFront_byAttribDoc* GetDocument() const;

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
	virtual ~CAddFront_byAttribView();
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

#ifndef _DEBUG  // AddFront_byAttribView.cpp의 디버그 버전
inline CAddFront_byAttribDoc* CAddFront_byAttribView::GetDocument() const
   { return reinterpret_cast<CAddFront_byAttribDoc*>(m_pDocument); }
#endif

