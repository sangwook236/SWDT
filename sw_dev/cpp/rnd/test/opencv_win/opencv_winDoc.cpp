// opencv_winDoc.cpp : Copencv_winDoc 클래스의 구현
//

#include "stdafx.h"
#include "opencv_win.h"

#include "opencv_winDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// Copencv_winDoc

IMPLEMENT_DYNCREATE(Copencv_winDoc, CDocument)

BEGIN_MESSAGE_MAP(Copencv_winDoc, CDocument)
END_MESSAGE_MAP()


// Copencv_winDoc 생성/소멸

Copencv_winDoc::Copencv_winDoc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

Copencv_winDoc::~Copencv_winDoc()
{
}

BOOL Copencv_winDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// Copencv_winDoc serialization

void Copencv_winDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}


// Copencv_winDoc 진단

#ifdef _DEBUG
void Copencv_winDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void Copencv_winDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// Copencv_winDoc 명령
