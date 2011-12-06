// opencv_winView.h : Copencv_winView 클래스의 인터페이스
//


#pragma once

#include "OpenCvVisionSensor.h"

class Copencv_winView : public CView
{
protected: // serialization에서만 만들어집니다.
	Copencv_winView();
	DECLARE_DYNCREATE(Copencv_winView)

// 특성입니다.
public:
	Copencv_winDoc* GetDocument() const;

// 작업입니다.
public:

private:
	OpenCvVisionSensor visionSensor_;

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);

// 구현입니다.
public:
	virtual ~Copencv_winView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnOpencvcamStartcam();
	afx_msg void OnOpencvcamStopcam();
	virtual void OnInitialUpdate();
};

#ifndef _DEBUG  // opencv_winView.cpp의 디버그 버전
inline Copencv_winDoc* Copencv_winView::GetDocument() const
   { return reinterpret_cast<Copencv_winDoc*>(m_pDocument); }
#endif

