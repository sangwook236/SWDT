// MicrosoftDoc.cpp : CMicrosoftDoc 클래스의 구현
//

#include "stdafx.h"
#include "Microsoft.h"

#include "MicrosoftDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMicrosoftDoc

IMPLEMENT_DYNCREATE(CMicrosoftDoc, CDocument)

BEGIN_MESSAGE_MAP(CMicrosoftDoc, CDocument)
END_MESSAGE_MAP()


// CMicrosoftDoc 생성/소멸

CMicrosoftDoc::CMicrosoftDoc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

CMicrosoftDoc::~CMicrosoftDoc()
{
}

BOOL CMicrosoftDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CMicrosoftDoc serialization

void CMicrosoftDoc::Serialize(CArchive& ar)
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


// CMicrosoftDoc 진단

#ifdef _DEBUG
void CMicrosoftDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CMicrosoftDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CMicrosoftDoc 명령
