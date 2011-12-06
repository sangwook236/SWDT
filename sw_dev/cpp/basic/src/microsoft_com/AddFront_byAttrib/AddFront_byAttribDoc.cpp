// AddFront_byAttribDoc.cpp : CAddFront_byAttribDoc 클래스의 구현
//

#include "stdafx.h"
#include "AddFront_byAttrib.h"

#include "AddFront_byAttribDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAddFront_byAttribDoc

IMPLEMENT_DYNCREATE(CAddFront_byAttribDoc, CDocument)

BEGIN_MESSAGE_MAP(CAddFront_byAttribDoc, CDocument)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(CAddFront_byAttribDoc, CDocument)
END_DISPATCH_MAP()

// 참고: IID_IAddFront_byAttrib에 대한 지원을 추가하여
//  VBA에서 형식 안전 바인딩을 지원합니다. 
//  이 IID는 .IDL 파일의 dispinterface에 첨부된 GUID와 일치해야 합니다.

// {A855185B-71B9-4228-87A8-90AC95A08C8F}
static const IID IID_IAddFront_byAttrib =
{ 0xA855185B, 0x71B9, 0x4228, { 0x87, 0xA8, 0x90, 0xAC, 0x95, 0xA0, 0x8C, 0x8F } };

BEGIN_INTERFACE_MAP(CAddFront_byAttribDoc, CDocument)
	INTERFACE_PART(CAddFront_byAttribDoc, IID_IAddFront_byAttrib, Dispatch)
END_INTERFACE_MAP()


// CAddFront_byAttribDoc 생성/소멸

CAddFront_byAttribDoc::CAddFront_byAttribDoc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

	EnableAutomation();

	AfxOleLockApp();
}

CAddFront_byAttribDoc::~CAddFront_byAttribDoc()
{
	AfxOleUnlockApp();
}

BOOL CAddFront_byAttribDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CAddFront_byAttribDoc serialization

void CAddFront_byAttribDoc::Serialize(CArchive& ar)
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


// CAddFront_byAttribDoc 진단

#ifdef _DEBUG
void CAddFront_byAttribDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CAddFront_byAttribDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CAddFront_byAttribDoc 명령
