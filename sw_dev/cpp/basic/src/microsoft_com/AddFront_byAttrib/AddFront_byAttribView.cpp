// AddFront_byAttribView.cpp : CAddFront_byAttribView 클래스의 구현
//

#include "stdafx.h"
#include "AddFront_byAttrib.h"

#include "AddFront_byAttribDoc.h"
#include "AddFront_byAttribView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAddFront_byAttribView

IMPLEMENT_DYNCREATE(CAddFront_byAttribView, CFormView)

BEGIN_MESSAGE_MAP(CAddFront_byAttribView, CFormView)
	ON_BN_CLICKED(IDC_BUTTON_ADD, &CAddFront_byAttribView::OnBnClickedButtonAdd)
	ON_BN_CLICKED(IDC_BUTTON_ADD_TEN, &CAddFront_byAttribView::OnBnClickedButtonAddTen)
	ON_BN_CLICKED(IDC_BUTTON_CLEAR, &CAddFront_byAttribView::OnBnClickedButtonClear)
END_MESSAGE_MAP()

// CAddFront_byAttribView 생성/소멸

CAddFront_byAttribView::CAddFront_byAttribView()
	: CFormView(CAddFront_byAttribView::IDD)
	, m_iSum(0)
	, m_iAddEnd(0)
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CAddFront_byAttribView::~CAddFront_byAttribView()
{
}

void CAddFront_byAttribView::DoDataExchange(CDataExchange* pDX)
{
	CFormView::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_SUM, m_iSum);
	DDX_Text(pDX, IDC_EDIT_ADD_END, m_iAddEnd);
}

BOOL CAddFront_byAttribView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CFormView::PreCreateWindow(cs);
}

void CAddFront_byAttribView::OnInitialUpdate()
{
	CFormView::OnInitialUpdate();
	GetParentFrame()->RecalcLayout();
	ResizeParentToFit();

#if defined(__USE_ADDBACK_DUAL) || defined(__USE_ADDBACK_DISPATCH)
	addBackPtr_ = IAddBackPtr(__uuidof(AddBack));
#elif defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	addBackPtr_ = IAddBackPtr(__uuidof(CAddBack));
#endif
	CComObject<CEventHandler>::CreateInstance(&handlerPtr_);

	if (!handlerPtr_->HookEvent(addBackPtr_))
	{
		MessageBox(_T("이벤트를 받을 수 없습니다."), _T("이벤트 에러"), MB_OK);
		return;
	}

	m_iAddEnd = addBackPtr_->AddEnd;
	m_iSum = addBackPtr_->Sum;
	UpdateData(FALSE);
}


// CAddFront_byAttribView 진단

#ifdef _DEBUG
void CAddFront_byAttribView::AssertValid() const
{
	CFormView::AssertValid();
}

void CAddFront_byAttribView::Dump(CDumpContext& dc) const
{
	CFormView::Dump(dc);
}

CAddFront_byAttribDoc* CAddFront_byAttribView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CAddFront_byAttribDoc)));
	return (CAddFront_byAttribDoc*)m_pDocument;
}
#endif //_DEBUG


// CAddFront_byAttribView 메시지 처리기

void CAddFront_byAttribView::OnBnClickedButtonAdd()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	UpdateData(TRUE);
	try
	{
		addBackPtr_->AddEnd = m_iAddEnd;
		addBackPtr_->Add();
		m_iSum = addBackPtr_->Sum;
	}
	catch(const _com_error &e)
	{
		_bstr_t bstrSource(e.Source());
		_bstr_t bstrDescription(e.Description());
		CString szMsg(_T("에러가 발생했습니다.!\n"));
		CString szTemp;

		szTemp.Format(_T("에러코드: %081x\n"), e.Error());
		szMsg += szTemp;
		szTemp.Format(_T("에러내용: %s\n"), e.ErrorMessage());
		szMsg += szTemp;
		szTemp.Format(_T("에러소스: %s\n"), bstrSource.length() ? (LPCTSTR)bstrSource : _T("없음"));
		szMsg += szTemp;
		MessageBox(szMsg.GetBuffer(szMsg.GetLength()), _T("에러 발생"));
	}

	UpdateData(FALSE);
}

void CAddFront_byAttribView::OnBnClickedButtonAddTen()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	try
	{
		addBackPtr_->AddTen();
		m_iSum = addBackPtr_->Sum;
	}
	catch(const _com_error &e)
	{
		_bstr_t bstrSource(e.Source());
		_bstr_t bstrDescription(e.Description());
		CString szMsg(_T("에러가 발생했습니다.!\n"));
		CString szTemp;

		szTemp.Format(_T("에러코드: %081x\n"), e.Error());
		szMsg += szTemp;
		szTemp.Format(_T("에러내용: %s\n"), e.ErrorMessage());
		szMsg += szTemp;
		szTemp.Format(_T("에러소스: %s\n"), bstrSource.length() ? (LPCTSTR)bstrSource : _T("없음"));
		szMsg += szTemp;
		MessageBox(szMsg.GetBuffer(szMsg.GetLength()), _T("에러 발생"));
	}

	UpdateData(FALSE);
}

void CAddFront_byAttribView::OnBnClickedButtonClear()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	try
	{
		addBackPtr_->Clear();
		m_iSum = addBackPtr_->Sum;
	}
	catch(const _com_error &e)
	{
		_bstr_t bstrSource(e.Source());
		_bstr_t bstrDescription(e.Description());
		CString szMsg(_T("에러가 발생했습니다.!\n"));
		CString szTemp;

		szTemp.Format(_T("에러코드: %081x\n"), e.Error());
		szMsg += szTemp;
		szTemp.Format(_T("에러내용: %s\n"), e.ErrorMessage());
		szMsg += szTemp;
		szTemp.Format(_T("에러소스: %s\n"), bstrSource.length() ? (LPCTSTR)bstrSource : _T("없음"));
		szMsg += szTemp;
		MessageBox(szMsg.GetBuffer(szMsg.GetLength()), _T("에러 발생"));
	}

	UpdateData(FALSE);
}

BOOL CAddFront_byAttribView::DestroyWindow()
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.
#if defined(__USE_ADDBACK_DUAL)
	//handlerPtr_->UnhookEvent(addBackPtr_);
	AtlUnadvise(addBackPtr_, __uuidof(IAddBackEvents), cookie_);
#elif defined(__USE_ADDBACK_DISPATCH) || defined(__USE_ADDBACK_BYATTRIB_DISPATCH)
	handlerPtr_->UnhookEvent(addBackPtr_);
#endif

	handlerPtr_ = 0;
	addBackPtr_ = 0;

	return CFormView::DestroyWindow();
}
