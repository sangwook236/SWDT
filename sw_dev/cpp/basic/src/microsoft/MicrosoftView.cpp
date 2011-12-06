// MicrosoftView.cpp : CMicrosoftView 클래스의 구현
//

#include "stdafx.h"
#include "Microsoft.h"

#include "MicrosoftDoc.h"
#include "MicrosoftView.h"

#include <mmsystem.h>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMicrosoftView

IMPLEMENT_DYNCREATE(CMicrosoftView, CView)

BEGIN_MESSAGE_MAP(CMicrosoftView, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(ID_SOUND_SNDPLAYSOUND, &CMicrosoftView::OnSoundSndplaysound)
	ON_COMMAND(ID_SOUND_PLAYSOUND, &CMicrosoftView::OnSoundPlaysound)
END_MESSAGE_MAP()

// CMicrosoftView 생성/소멸

CMicrosoftView::CMicrosoftView()
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CMicrosoftView::~CMicrosoftView()
{
}

BOOL CMicrosoftView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// CMicrosoftView 그리기

void CMicrosoftView::OnDraw(CDC* /*pDC*/)
{
	CMicrosoftDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
}


// CMicrosoftView 인쇄

BOOL CMicrosoftView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void CMicrosoftView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void CMicrosoftView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}


// CMicrosoftView 진단

#ifdef _DEBUG
void CMicrosoftView::AssertValid() const
{
	CView::AssertValid();
}

void CMicrosoftView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMicrosoftDoc* CMicrosoftView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMicrosoftDoc)));
	return (CMicrosoftDoc*)m_pDocument;
}
#endif //_DEBUG


// CMicrosoftView 메시지 처리기

void CMicrosoftView::OnSoundSndplaysound()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	//sndPlaySound(_T(".\\data\\sound\\wmpaud1.wav"), SND_ASYNC);
	sndPlaySound(_T(".\\data\\sound\\oasis.wav"), SND_ASYNC);
	Sleep(1000);
	sndPlaySound(_T(".\\data\\sound\\wmpaud6.wav"), SND_ASYNC);
}

void CMicrosoftView::OnSoundPlaysound()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	PlaySound(_T(".\\data\\sound\\oasis.wav"), NULL, SND_FILENAME);
	Sleep(1000);
	PlaySound(_T(".\\data\\sound\\wmpaud6.wav"), NULL, SND_FILENAME);
}
