// AddFrontDoc.cpp : CAddFrontDoc 클래스의 구현
//

#include "stdafx.h"
#include "AddFront.h"

#include "AddFrontDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAddFrontDoc

IMPLEMENT_DYNCREATE(CAddFrontDoc, CDocument)

BEGIN_MESSAGE_MAP(CAddFrontDoc, CDocument)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(CAddFrontDoc, CDocument)
END_DISPATCH_MAP()

// 참고: IID_IAddFront에 대한 지원을 추가하여
//  VBA에서 형식 안전 바인딩을 지원합니다. 
//  이 IID는 .IDL 파일의 dispinterface에 첨부된 GUID와 일치해야 합니다.

// {75575FAA-B604-48F5-A0DD-5F0674D05352}
static const IID IID_IAddFront =
{ 0x75575FAA, 0xB604, 0x48F5, { 0xA0, 0xDD, 0x5F, 0x6, 0x74, 0xD0, 0x53, 0x52 } };

BEGIN_INTERFACE_MAP(CAddFrontDoc, CDocument)
	INTERFACE_PART(CAddFrontDoc, IID_IAddFront, Dispatch)
END_INTERFACE_MAP()


// CAddFrontDoc 생성/소멸

CAddFrontDoc::CAddFrontDoc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

	EnableAutomation();

	AfxOleLockApp();
}

CAddFrontDoc::~CAddFrontDoc()
{
	AfxOleUnlockApp();
}

BOOL CAddFrontDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CAddFrontDoc serialization

void CAddFrontDoc::Serialize(CArchive& ar)
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


// CAddFrontDoc 진단

#ifdef _DEBUG
void CAddFrontDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CAddFrontDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CAddFrontDoc 명령
