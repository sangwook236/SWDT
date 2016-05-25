// opencv_winView.cpp : Copencv_winView 클래스의 구현
//

#include "stdafx.h"
#include "opencv_win.h"

#include "opencv_winDoc.h"
#include "opencv_winView.h"

#include <opencv/highgui.h>
//#include <opencv/cvcam.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// Copencv_winView

IMPLEMENT_DYNCREATE(Copencv_winView, CView)

BEGIN_MESSAGE_MAP(Copencv_winView, CView)
	// 표준 인쇄 명령입니다.
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_COMMAND(ID_OPENCVCAM_STARTCAM, &Copencv_winView::OnOpencvcamStartcam)
	ON_COMMAND(ID_OPENCVCAM_STOPCAM, &Copencv_winView::OnOpencvcamStopcam)
END_MESSAGE_MAP()

// Copencv_winView 생성/소멸

Copencv_winView::Copencv_winView()
: visionSensor_(320, 240)
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

Copencv_winView::~Copencv_winView()
{
}

BOOL Copencv_winView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}

// Copencv_winView 그리기

void Copencv_winView::OnDraw(CDC* /*pDC*/)
{
	Copencv_winDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	// TODO: 여기에 원시 데이터에 대한 그리기 코드를 추가합니다.
}


// Copencv_winView 인쇄

BOOL Copencv_winView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// 기본적인 준비
	return DoPreparePrinting(pInfo);
}

void Copencv_winView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄하기 전에 추가 초기화 작업을 추가합니다.
}

void Copencv_winView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: 인쇄 후 정리 작업을 추가합니다.
}


// Copencv_winView 진단

#ifdef _DEBUG
void Copencv_winView::AssertValid() const
{
	CView::AssertValid();
}

void Copencv_winView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

Copencv_winDoc* Copencv_winView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(Copencv_winDoc)));
	return (Copencv_winDoc*)m_pDocument;
}
#endif //_DEBUG


// Copencv_winView 메시지 처리기

void Copencv_winView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
}

void Copencv_winView::OnOpencvcamStartcam()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	if (!visionSensor_.isInitialized())
	{
		const int camCount = cvcamGetCamerasCount();
		if (0 == camCount)
		{
			AfxMessageBox(_T("available camera not found"), MB_OK | MB_ICONSTOP);
			return;
		}
		const int camId = 0;
/*
		int* selectedCamIndexes;
		const int selectedCamCount = cvcamSelectCamera(&selectedCamIndexes);
		if (0 == selectedCamCount)
		{
			AfxMessageBox(_T("any cam failed to be connected"), MB_OK | MB_ICONSTOP);
			return;
		}
		const int camId = selectedCamIndexes[0];
*/
		visionSensor_.setSensorId(camId);
		visionSensor_.setWindowHandle((void*)&m_hWnd);

		visionSensor_.initSystem();
	}

	visionSensor_.startCapturing();
}

void Copencv_winView::OnOpencvcamStopcam()
{
	// TODO: 여기에 명령 처리기 코드를 추가합니다.
	visionSensor_.stopCapturing();
	visionSensor_.reset();
}
